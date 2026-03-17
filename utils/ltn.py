from itertools import combinations
from typing import Dict, Iterable, List

import torch

EPS = 1e-10


class LTNRuleManager:
    """Evaluate grounded logical rules with differentiable fuzzy semantics."""

    def __init__(self, eps: float = EPS):
        self.eps = eps

    @staticmethod
    def fuzzy_not(value: torch.Tensor) -> torch.Tensor:
        return 1.0 - value

    @staticmethod
    def fuzzy_and(values: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(values)
        if not values:
            return torch.tensor(1.0)
        stacked = torch.stack(values).reshape(len(values), -1)
        return torch.prod(stacked, dim=0).squeeze()

    @staticmethod
    def fuzzy_or(values: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(values)
        if not values:
            return torch.tensor(0.0)
        stacked = torch.stack(values).reshape(len(values), -1)
        return (1.0 - torch.prod(1.0 - stacked, dim=0)).squeeze()

    @staticmethod
    def fuzzy_implies(antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        return (1.0 - antecedent + antecedent * consequent).clamp(0.0, 1.0)

    @staticmethod
    def fuzzy_equivalence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.abs(left - right)).clamp(0.0, 1.0)

    def _literal_truth(self, literal: dict, statement_truths: Dict[str, torch.Tensor]) -> torch.Tensor:
        truth = statement_truths[literal["statement"]]
        truth = truth.squeeze()
        return truth if literal.get("positive", True) else self.fuzzy_not(truth)

    @staticmethod
    def _belief_truth(literal: dict, belief_lookup: Dict[str, Dict[str, int]], subject: str) -> torch.Tensor:
        subject_beliefs = belief_lookup.get(subject, {})
        belief = float(subject_beliefs.get(literal["predicate"], 0))
        value = torch.tensor(belief)
        return value if literal.get("positive", True) else 1.0 - value

    def evaluate_rule(self, rule: dict, statement_truths: Dict[str, torch.Tensor]) -> torch.Tensor:
        rule_type = rule["rule_type"]

        if rule_type in {"universal", "and_implies"}:
            antecedent = self.fuzzy_and(
                self._literal_truth(literal, statement_truths) for literal in rule["antecedents"]
            )
            consequent = self.fuzzy_and(
                self._literal_truth(literal, statement_truths) for literal in rule["consequents"]
            )
            return self.fuzzy_implies(antecedent, consequent)

        if rule_type == "or_implies":
            antecedent = self.fuzzy_or(
                self._literal_truth(literal, statement_truths) for literal in rule["antecedents"]
            )
            consequent = self.fuzzy_and(
                self._literal_truth(literal, statement_truths) for literal in rule["consequents"]
            )
            return self.fuzzy_implies(antecedent, consequent)

        if rule_type == "equivalence":
            left = self.fuzzy_and(
                self._literal_truth(literal, statement_truths) for literal in rule["antecedents"]
            )
            right = self.fuzzy_and(
                self._literal_truth(literal, statement_truths) for literal in rule["consequents"]
            )
            return self.fuzzy_equivalence(left, right)

        if rule_type == "xor":
            atoms = [self._literal_truth(literal, statement_truths) for literal in rule["atoms"]]
            if len(atoms) < 2:
                return torch.tensor(1.0)
            pairwise = [self.fuzzy_not(self.fuzzy_and((left, right))) for left, right in combinations(atoms, 2)]
            return self.fuzzy_and(pairwise)

        raise ValueError(f"Unsupported rule type: {rule_type}")

    def batch_rule_truths(self, batch: List[dict], statement_truths: Dict[str, torch.Tensor]) -> torch.Tensor:
        truths = [self.evaluate_rule(rule=sample, statement_truths=statement_truths) for sample in batch]
        if not truths:
            return torch.empty(0)
        return torch.stack(truths).reshape(-1)

    def logic_loss(self, truths: torch.Tensor) -> torch.Tensor:
        if truths.numel() == 0:
            return torch.tensor(0.0)
        return -torch.log(truths.clamp_min(self.eps)).mean()

    def average_satisfaction(self, grounded_rules: List[dict], belief_lookup: Dict[str, Dict[str, int]]) -> float:
        if not grounded_rules:
            return 0.0

        truths = []
        for rule in grounded_rules:
            subject = rule["subject"]
            statement_truths = {}
            for literal in rule["antecedents"] + rule["consequents"] + rule["atoms"]:
                key = literal["statement"]
                if key not in statement_truths:
                    statement_truths[key] = self._belief_truth(literal, belief_lookup, subject)
            truths.append(self.evaluate_rule(rule, statement_truths).item())

        return sum(truths) / len(truths)
