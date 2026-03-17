# LTN Training Pipeline

## Current Baseline

The existing training path keeps logic outside the model as an SDD-based semantic loss:

1. `run.py` loads a config, instantiates `QA`, `ConstraintManager`, and `Trainer`.
2. `dataset/beliefbank.py` loads `constraints_v2.json`, `calibration_facts.json`, and `silver_facts.json`.
3. Facts are converted from symbolic predicates such as `Relation,Object` into natural-language statements.
4. Binary constraints are grounded per subject into antecedent/consequent statement pairs.
5. `QA.get_formula_beliefs()` scores the antecedent and consequent statements with the LM.
6. `ConstraintManager.sl()` compiles each grounded rule into an SDD and computes semantic loss from the probability mass of satisfying assignments.
7. `Trainer.epoch()` backpropagates only that logic loss.

## LTN Backend

The new backend keeps the LM as the fact scorer and replaces the symbolic loss with differentiable fuzzy rule satisfaction:

1. `run.py` selects `logic_backend=ltn` and passes rule settings into `Trainer`.
2. `dataset/beliefbank.py` normalizes both legacy constraints and augmented rules from `scripts/generate_rules_llm.py` into a shared rule schema.
3. Rules are grounded per subject into LTN-ready literals with natural-language fact statements.
4. `QA.get_fact_probabilities()` exposes a reusable predicate interface for statement truth scores.
5. `utils/ltn.py` evaluates normalized rules with fuzzy `and`, `or`, `not`, implication, equivalence, and XOR semantics.
6. `Trainer.epoch_ltn()` combines factual supervision on observed facts with rule-satisfaction loss over grounded formulas.
7. Evaluation reports factual quality plus average grounded rule satisfaction.

## Rule Schema

All LTN rules are normalized into this structure:

```python
{
    "id": "rule id",
    "source": "legacy|augmented",
    "rule_type": "universal|equivalence|and_implies|or_implies|xor",
    "antecedents": [{"predicate": "IsA,dog", "positive": True}],
    "consequents": [{"predicate": "IsA,mammal", "positive": True}],
    "atoms": []
}
```

## Recommended Config

Use the new backend with:

```json
{
  "logic_backend": "ltn",
  "rule_source": "hybrid",
  "logic_weight": 1.0,
  "factual_weight": 1.0
}
```
