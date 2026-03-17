from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTS_PATH = REPO_ROOT / "data" / "beliefbank" / "silver_facts.json"
DEFAULT_CONSTRAINTS_PATH = REPO_ROOT / "data" / "beliefbank" / "constraints_v2.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "beliefbank" / "beliefbank_augmented.json"


def load_beliefbank(facts_path: Path, constraints_path: Path):
    with open(facts_path, encoding="utf-8") as f:
        facts = json.load(f)
    with open(constraints_path, encoding="utf-8") as f:
        constraints = json.load(f)
    return facts, constraints


def extract_relations_and_entities(facts: dict) -> tuple[dict[str, set[str]], list[str]]:
    relation_to_true_entities: dict[str, set[str]] = {}
    for entity, relation_dict in facts.items():
        for relation, label in relation_dict.items():
            if relation not in relation_to_true_entities:
                relation_to_true_entities[relation] = set()
            if label == "yes":
                relation_to_true_entities[relation].add(entity)
    all_entities = list(facts.keys())
    return relation_to_true_entities, all_entities


def validate_rule(
    rule: dict,
    relation_to_true_entities: dict[str, set[str]],
    all_entities: list[str],
) -> bool:
    if not isinstance(rule, dict) or "type" not in rule:
        return False

    rtype = rule.get("type")
    if rtype == "xor":
        atoms = rule.get("atoms", [])
        if len(atoms) < 2:
            return False
        for entity in all_entities:
            true_count = sum(
                1
                for a in atoms
                if entity in relation_to_true_entities.get(a, set())
            )
            if true_count > 1:
                return False
        return True

    if rtype == "and_implies":
        antecedents = rule.get("antecedents", [])
        consequent = rule.get("consequent")
        if not antecedents or not consequent:
            return False
        consequent_entities = relation_to_true_entities.get(consequent, set())
        for entity in all_entities:
            all_ante_true = all(
                entity in relation_to_true_entities.get(a, set())
                for a in antecedents
            )
            if all_ante_true and entity not in consequent_entities:
                return False
        return True

    if rtype == "equivalence":
        left = rule.get("left")
        right = rule.get("right")
        if not left or not right:
            return False
        left_entities = relation_to_true_entities.get(left, set())
        right_entities = relation_to_true_entities.get(right, set())
        return left_entities == right_entities

    if rtype == "universal":
        antecedent = rule.get("antecedent")
        consequent = rule.get("consequent")
        if not antecedent or not consequent:
            return False
        ante_entities = relation_to_true_entities.get(antecedent, set())
        cons_entities = relation_to_true_entities.get(consequent, set())
        return ante_entities.issubset(cons_entities)

    if rtype == "or_implies":
        antecedents = rule.get("antecedents", [])
        consequent = rule.get("consequent")
        if len(antecedents) < 2 or not consequent:
            return False
        or_entities = set()
        for a in antecedents:
            or_entities |= relation_to_true_entities.get(a, set())
        cons_entities = relation_to_true_entities.get(consequent, set())
        return or_entities.issubset(cons_entities)

    return False


def parse_json_array_from_text(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        start = text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        return []


def _openai_generate(model_name: str, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env")
    client = OpenAI(api_key=api_key)

    last_exc = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_429 = "429" in err_str or "rate limit" in err_str or "quota" in err_str
            if is_429 and attempt < 3:
                wait = 35 * (attempt + 1)
                print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 2}/4...")
                time.sleep(wait)
            else:
                raise
    raise last_exc


def generate_rules_with_llm(facts: dict, relations: list[str], model: str = "gpt-4o-mini") -> list[dict]:
    relations_str = "\n".join(f"- {r}" for r in relations)

    all_rules: list[dict] = []
    context = ""

    user1 = f"""You are helping build a logic dataset for animals and plants.
Here are the available relations in the dataset (use these exact strings):
{relations_str}

We have 85 entities with yes/no labels over these relations. You will not see the raw facts, but every rule you propose will be strictly checked against them.

Generate 200 XOR rules: groups of relations where an entity can be TRUE for AT MOST ONE.
Example: an animal cannot be both a mammal and a bird -> {{"type": "xor", "atoms": ["IsA,mammal", "IsA,bird"]}}.

Respond ONLY with a single JSON array. No explanation, no backticks, no extra text. The first character of your response must be [ and the last character must be ].
Format:
[
  {{"type": "xor", "atoms": ["relation1", "relation2", "relation3"]}},
  ...
]
Use only relations from the list above (exact spelling)."""
    msg1 = _openai_generate(model, context + user1 if context else user1)
    context += f"User:\n{user1}\n\nAssistant:\n{msg1}\n\n"
    parsed = parse_json_array_from_text(msg1)
    for r in parsed:
        if isinstance(r, dict) and r.get("type") == "xor":
            all_rules.append(r)
    if not parsed:
        print("Warning: could not parse XOR rules response")

    user2 = """Generate 200 conjunction-implication rules (and_implies): two or more conditions that together imply a third.
Example: if an entity HasA,fur AND HasProperty,warm-blooded -> IsA,mammal.

Respond ONLY with a single JSON array. No explanation, no backticks, no extra text. The first character of your response must be [ and the last character must be ].
Format:
[
  {"type": "and_implies", "antecedents": ["relation1", "relation2"], "consequent": "relation3"},
  ...
]
Use only relations from the original list (exact strings)."""
    msg2 = _openai_generate(model, context + "User:\n" + user2 + "\n\nAssistant:\n")
    context += f"User:\n{user2}\n\nAssistant:\n{msg2}\n\n"
    parsed = parse_json_array_from_text(msg2)
    for r in parsed:
        if isinstance(r, dict) and r.get("type") == "and_implies":
            all_rules.append(r)
    if not parsed:
        print("Warning: could not parse and_implies rules response")

    user3 = """Generate:
- 200 equivalence rules: two relations that are always true/false together (same entities).
- 200 universal rules: if relation A is true for an entity, relation B must also be true for that entity.

Respond ONLY with a single JSON array. No explanation, no backticks, no extra text. The first character of your response must be [ and the last character must be ].
Format:
[
  {"type": "equivalence", "left": "relation1", "right": "relation2"},
  {"type": "universal", "antecedent": "relation1", "consequent": "relation2"},
  ...
]
Use only relations from the original list (exact strings)."""
    msg3 = _openai_generate(model, context + "User:\n" + user3 + "\n\nAssistant:\n")
    context += f"User:\n{user3}\n\nAssistant:\n{msg3}\n\n"
    parsed = parse_json_array_from_text(msg3)
    for r in parsed:
        if isinstance(r, dict) and r.get("type") in ("equivalence", "universal"):
            all_rules.append(r)
    if not parsed:
        print("Warning: could not parse equivalence/universal rules response")

    user4 = """Generate 200 disjunction-implication rules (or_implies): if an entity has AT LEAST ONE of the given relations, it must also have the consequent.
Example: if something IsA,dog OR IsA,cat -> IsA,mammal. So {"type": "or_implies", "antecedents": ["IsA,dog", "IsA,cat"], "consequent": "IsA,mammal"}.

Respond ONLY with a single JSON array. No explanation, no backticks, no extra text. The first character of your response must be [ and the last character must be ].
Format:
[
  {"type": "or_implies", "antecedents": ["relation1", "relation2", "..."], "consequent": "relation3"},
  ...
]
Use only relations from the original list (exact strings). Need at least 2 antecedents per rule."""
    msg4 = _openai_generate(model, context + "User:\n" + user4 + "\n\nAssistant:\n")
    parsed = parse_json_array_from_text(msg4)
    for r in parsed:
        if isinstance(r, dict) and r.get("type") == "or_implies":
            all_rules.append(r)
    if not parsed:
        print("Warning: could not parse or_implies rules response")

    return all_rules


def augment_beliefbank(
    facts_path: Path,
    constraints_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
) -> None:
    facts, original_constraints = load_beliefbank(facts_path, constraints_path)
    relation_to_true_entities, all_entities = extract_relations_and_entities(facts)
    all_relations = list(relation_to_true_entities.keys())

    print(f"Loaded {len(all_relations)} unique relations across {len(all_entities)} entities.")

    if not (os.environ.get("OPENAI_API_KEY")):
        print("Warning: OPENAI_API_KEY not set. Set in .env.")
    print("Generating candidate rules with LLM...")
    candidate_rules = generate_rules_with_llm(facts, all_relations, model=model)

    print(f"Generated {len(candidate_rules)} candidate rules.")

    valid_rules: list[dict] = []
    seen_keys: set[str] = set()
    for rule in candidate_rules:
        if not validate_rule(rule, relation_to_true_entities, all_entities):
            print(f"  [FAIL] Invalid: {rule}")
            continue
        key = json.dumps(rule, sort_keys=True)
        if key in seen_keys:
            print(f"  [DUP]  Skip duplicate: {rule}")
            continue
        seen_keys.add(key)
        valid_rules.append(rule)
        print(f"  [OK]   Valid:   {rule}")

    print(f"\n{len(valid_rules)}/{len(candidate_rules)} rules passed validation (after deduplication).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented = {
        "facts": facts,
        "original_constraints": original_constraints,
        "augmented_constraints": valid_rules,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)
    print(f"Saved to {output_path}")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Generate and validate LLM rules for BeliefBank.")
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS_PATH, help="Path to silver_facts.json")
    parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS_PATH, help="Path to constraints graph JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output augmented JSON path")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    args = parser.parse_args()

    augment_beliefbank(
        facts_path=args.facts,
        constraints_path=args.constraints,
        output_path=args.output,
        model=args.model,
    )


if __name__ == "__main__":
    main()
