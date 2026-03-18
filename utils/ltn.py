from itertools import combinations
from typing import Dict, Iterable, List

import torch


EPS = 1e-6


class LTNRuleManager:
    """
    Evaluate grounded logical rules with differentiable fuzzy semantics.

    Follows the LTNtorch "stable product configuration":
      - Product t-norm for AND
      - Probabilistic sum for OR
      - Reichenbach implication (with configurable floor)
      - Standard negation
      - pMeanError aggregation for the universal quantifier / loss
    All operators apply stability clamping to avoid vanishing / exploding
    gradients at the edges of [0, 1].
    """

    def __init__(
        self,
        eps: float = EPS,
        implication_floor: float = 0.5,
        p: float = 2.0,
    ):
        self.eps = eps
        self.implication_floor = implication_floor
        self.p = p

    def _stable(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.eps, 1.0 - self.eps)

    # ------------------------------------------------------------------
    # Connectives  (product configuration with stability)
    # ------------------------------------------------------------------

    @staticmethod
    def fuzzy_not(value: torch.Tensor) -> torch.Tensor:
        return 1.0 - value

    def fuzzy_and(self, values: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(values)
        if not values:
            return torch.tensor(1.0)
        stacked = self._stable(torch.stack(values).reshape(len(values), -1))
        return self._stable(torch.prod(stacked, dim=0)).squeeze()

    def fuzzy_or(self, values: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(values)
        if not values:
            return torch.tensor(0.0)
        stacked = self._stable(torch.stack(values).reshape(len(values), -1))
        return self._stable(1.0 - torch.prod(1.0 - stacked, dim=0)).squeeze()

    def fuzzy_implies(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Reichenbach-style implication with a configurable floor.

        Standard Reichenbach: 1 - A + A*C.  When the antecedent is near 0
        the implication is vacuously true (~1.0), providing no gradient.
        The *floor* parameter replaces that regime so that inactive rules
        still produce a non-trivial truth value and therefore training
        pressure.

            truth = A * C + (1 - A) * floor
        """
        return self._stable(
            antecedent * consequent + (1.0 - antecedent) * self.implication_floor
        )

    @staticmethod
    def fuzzy_equivalence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.abs(left - right)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Literal / belief helpers
    # ------------------------------------------------------------------

    def _literal_truth(self, literal: dict, statement_truths: Dict[str, torch.Tensor]) -> torch.Tensor:
        truth = self._stable(statement_truths[literal["statement"]].squeeze())
        return truth if literal.get("positive", True) else self.fuzzy_not(truth)

    @staticmethod
    def _belief_truth(literal: dict, belief_lookup: Dict[str, Dict[str, int]], subject: str) -> torch.Tensor:
        subject_beliefs = belief_lookup.get(subject, {})
        belief = float(subject_beliefs.get(literal["predicate"], 0))
        value = torch.tensor(belief)
        return value if literal.get("positive", True) else 1.0 - value

    # ------------------------------------------------------------------
    # Rule applicability guard
    # ------------------------------------------------------------------

    @staticmethod
    def is_rule_applicable(rule: dict) -> bool:
        rule_type = rule["rule_type"]

        def positive_belief(literal: dict) -> bool:
            return literal.get("belief", -1) == 1

        if rule_type in {"universal", "and_implies"}:
            return all(positive_belief(lit) for lit in rule["antecedents"])
        if rule_type == "or_implies":
            return any(positive_belief(lit) for lit in rule["antecedents"])
        if rule_type == "equivalence":
            return any(positive_belief(lit) for lit in rule["antecedents"] + rule["consequents"])
        if rule_type == "xor":
            return sum(positive_belief(lit) for lit in rule["atoms"]) > 0
        return True

    # ------------------------------------------------------------------
    # Per-rule evaluation
    # ------------------------------------------------------------------

    def evaluate_rule(self, rule: dict, statement_truths: Dict[str, torch.Tensor]) -> torch.Tensor:
        rule_type = rule["rule_type"]

        if rule_type in {"universal", "and_implies"}:
            antecedent = self.fuzzy_and(
                self._literal_truth(lit, statement_truths) for lit in rule["antecedents"]
            )
            consequent = self.fuzzy_and(
                self._literal_truth(lit, statement_truths) for lit in rule["consequents"]
            )
            return self.fuzzy_implies(antecedent, consequent)

        if rule_type == "or_implies":
            antecedent = self.fuzzy_or(
                self._literal_truth(lit, statement_truths) for lit in rule["antecedents"]
            )
            consequent = self.fuzzy_and(
                self._literal_truth(lit, statement_truths) for lit in rule["consequents"]
            )
            return self.fuzzy_implies(antecedent, consequent)

        if rule_type == "equivalence":
            left = self.fuzzy_and(
                self._literal_truth(lit, statement_truths) for lit in rule["antecedents"]
            )
            right = self.fuzzy_and(
                self._literal_truth(lit, statement_truths) for lit in rule["consequents"]
            )
            return self.fuzzy_equivalence(left, right)

        if rule_type == "xor":
            atoms = [self._literal_truth(lit, statement_truths) for lit in rule["atoms"]]
            if len(atoms) < 2:
                return self._stable(torch.tensor(1.0))
            atom_sum = torch.stack(atoms).sum()
            return self._stable((1.0 - torch.relu(atom_sum - 1.0)).clamp(0.0, 1.0))

        raise ValueError(f"Unsupported rule type: {rule_type}")

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def batch_rule_truths(
        self,
        batch: List[dict],
        statement_truths: Dict[str, torch.Tensor],
        filter_applicable: bool = True,
    ) -> torch.Tensor:
        truths = [
            self.evaluate_rule(rule=sample, statement_truths=statement_truths)
            for sample in batch
            if (not filter_applicable or self.is_rule_applicable(sample))
        ]
        if not truths:
            return torch.empty(0)
        return torch.stack(truths).reshape(-1)

    # ------------------------------------------------------------------
    # Loss / aggregation
    # ------------------------------------------------------------------

    def p_mean_error(self, truths: torch.Tensor) -> torch.Tensor:
        """pMeanError aggregator (LTN standard for ForAll).

        pME(u_1 ... u_n) = 1 - ( (1/n) * sum( (1 - u_i)^p ) ) ^ (1/p)

        Focuses gradient on the hardest-to-satisfy rules via the power p.
        p=1  -> linear mean of errors (lenient)
        p=2  -> balanced (default, recommended for learning)
        p>=4 -> strict, approaches min (risk of single-passing gradients)
        """
        errors = (1.0 - truths).clamp_min(0.0) ** self.p
        agg = 1.0 - errors.mean() ** (1.0 / self.p)
        return self._stable(agg)

    def logic_loss(self, truths: torch.Tensor) -> torch.Tensor:
        if truths.numel() == 0:
            return torch.tensor(0.0)
        sat = self.p_mean_error(truths)
        return -torch.log(sat)

    # ------------------------------------------------------------------
    # Evaluation-time satisfaction (uses ground beliefs, not model probs)
    # ------------------------------------------------------------------

    def average_satisfaction(self, grounded_rules: List[dict], belief_lookup: Dict[str, Dict[str, int]]) -> float:
        if not grounded_rules:
            return 0.0

        truths = []
        for rule in grounded_rules:
            if not self.is_rule_applicable(rule):
                continue
            subject = rule["subject"]
            statement_truths: Dict[str, torch.Tensor] = {}
            for literal in rule["antecedents"] + rule["consequents"] + rule["atoms"]:
                key = literal["statement"]
                if key not in statement_truths:
                    statement_truths[key] = self._belief_truth(literal, belief_lookup, subject)
            truths.append(self.evaluate_rule(rule, statement_truths).item())

        return (sum(truths) / len(truths)) if truths else 0.0
