## LLM-generated rules (`scripts/generate_rules_llm.py`)

This repository includes a small rule-augmentation pipeline built on top of the original BeliefBank data.

- The script `scripts/generate_rules_llm.py` loads `silver_facts.json` (85 entities, ~12.6k labelled facts) and `constraints_v2.json` (the original constraint graph).
- It extracts all 601 relation strings and the list of entities, then queries a Gemini model in four stages to propose:
  - `xor` rules (at most one relation in a set can be true for a given entity),
  - `and_implies` rules (a conjunction of relations implies another relation),
  - `equivalence` rules (two relations are true/false together on exactly the same set of entities),
  - `universal` rules (if relation A holds for an entity, relation B must also hold),
  - `or_implies` rules (if an entity has at least one relation from a set, it must also have the consequent relation).
- Each candidate rule is validated against `silver_facts.json` and retained only if it holds for all 85 entities and is not a duplicate.

### Running the rule generator

From the repository root:

```bash
pip install -r requirements.txt
python scripts/generate_rules_llm.py --model gemini-2.5-flash-lite
```

The script expects a `GEMINI_API_KEY` in `.env` in the repo root (see `.env.example` for the key name). On success it writes:

- `beliefbank_augmented.json`, a JSON object with:
  - `facts`: a copy of `silver_facts.json`,
  - `original_constraints`: a copy of `constraints_v2.json`,
  - `augmented_constraints`: the set of validated LLM-generated rules together with a small number of manually specified seed rules (covering the same rule types: `xor`, `and_implies`, `equivalence`, `universal`, `or_implies`).
