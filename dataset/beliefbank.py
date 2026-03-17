import json
import os
import random
from typing import Dict, List

import torch
import torch.utils.data as data

from models.loco.prompts import *
from .dataset import *

DEFAULT_CONSTRAINTS_PATH = os.path.join("data", "beliefbank", "constraints_v2.json")
DEFAULT_CALIBRATION_FACTS_PATH = os.path.join("data", "beliefbank", "calibration_facts.json")
DEFAULT_SILVER_FACTS_PATH = os.path.join("data", "beliefbank", "silver_facts.json")
DEFAULT_AUGMENTED_RULES_PATH = os.path.join("data", "beliefbank", "beliefbank_augmented.json")


def get_dataset(model_type: str, config: dict | None = None):
    """ Load and parse from file to torch.Dataset """
    config = config or {}
    constraints_path = config.get("constraints_path", DEFAULT_CONSTRAINTS_PATH)
    calibration_facts_path = config.get("calibration_facts_path", DEFAULT_CALIBRATION_FACTS_PATH)
    silver_facts_path = config.get("silver_facts_path", DEFAULT_SILVER_FACTS_PATH)
    rules_path = config.get("rules_path", DEFAULT_AUGMENTED_RULES_PATH)
    rule_source = config.get("rule_source", "legacy")
    logic_backend = config.get("logic_backend", "sdd")

    # Data loading
    constraints = Constraints(constraints_path=constraints_path, model_type=model_type)
    silver_facts = Facts(constraints=constraints, facts_path=silver_facts_path, model_type=model_type)
    calibration_facts = Facts(constraints=constraints, facts_path=calibration_facts_path, model_type=model_type)
    silver_splits = silver_facts.get_splits()
    calibration_splits = calibration_facts.get_splits()

    # Train
    train_constraints = TorchDataset(constraints.get_grounded_constraints(facts=calibration_splits["train"], path=constraints_path))
    train_calibration_facts = TorchDataset(calibration_splits["train"])
    train_silver_facts = TorchDataset(silver_splits["train"])
    # Val
    val_constraints = TorchDataset(constraints.get_grounded_constraints(facts=calibration_splits["val"], path=constraints_path))
    val_calibration_facts = TorchDataset(calibration_splits["val"])
    # Test
    test_calibration_facts = TorchDataset(calibration_splits["test"])
    test_silver_facts = TorchDataset(silver_splits["test"])

    data = {
        "constraints": {
            "train": train_constraints, # grounded
            "all": constraints, # set of links
        },
        "facts": {
            "calibration": {
                "train": train_calibration_facts,
                "test": test_calibration_facts,
                "val": val_calibration_facts,
                "complete": TorchDataset(calibration_facts.get_whole_set()),
            },
            "silver": {
                "train": train_silver_facts,
                "test": test_silver_facts,
                "complete": TorchDataset(silver_facts.get_whole_set())
            } 
        }
    }

    if logic_backend == "ltn":
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"),
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"),
        )
        normalized_rules = get_normalized_rules(
            constraints_path=constraints_path,
            rules_path=rules_path,
            rule_source=rule_source,
        )
        data["rules"] = {
            "all": normalized_rules,
            "train": TorchDataset(
                ground_rules(
                    rules=normalized_rules,
                    facts=calibration_splits["train"],
                    templates=templates,
                    uncountables=uncountables,
                )
            ),
            "val": {
                "calibration": ground_rules(
                    rules=normalized_rules,
                    facts=calibration_splits["val"],
                    templates=templates,
                    uncountables=uncountables,
                ),
            },
            "test": {
                "calibration": ground_rules(
                    rules=normalized_rules,
                    facts=calibration_facts.get_whole_set(),
                    templates=templates,
                    uncountables=uncountables,
                ),
                "silver": ground_rules(
                    rules=normalized_rules,
                    facts=silver_facts.get_whole_set(),
                    templates=templates,
                    uncountables=uncountables,
                ),
            },
        }

    return data


def normalize_literal(predicate: str, positive: bool = True) -> dict:
    return {
        "predicate": predicate,
        "positive": bool(positive),
    }


def normalize_legacy_rules(constraints_path: str) -> List[dict]:
    normalized = []
    for idx, rule in enumerate(Constraints.get_links(constraints_path)):
        normalized.append(
            {
                "id": f"legacy::{idx}",
                "source": "legacy",
                "rule_type": "universal",
                "antecedents": [normalize_literal(rule["antecedent"], rule["s_antecedent"])],
                "consequents": [normalize_literal(rule["consequent"], rule["s_consequent"])],
                "atoms": [],
            }
        )
    return normalized


def normalize_augmented_rules(rules_path: str) -> List[dict]:
    if not os.path.exists(rules_path):
        return []

    with open(rules_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    augmented_rules = payload.get("augmented_constraints", [])
    normalized = []
    for idx, rule in enumerate(augmented_rules):
        rule_type = rule["type"]
        sample = {
            "id": f"augmented::{idx}",
            "source": "augmented",
            "rule_type": rule_type,
            "antecedents": [],
            "consequents": [],
            "atoms": [],
        }
        if rule_type == "universal":
            sample["antecedents"] = [normalize_literal(rule["antecedent"])]
            sample["consequents"] = [normalize_literal(rule["consequent"])]
        elif rule_type == "equivalence":
            sample["antecedents"] = [normalize_literal(rule["left"])]
            sample["consequents"] = [normalize_literal(rule["right"])]
        elif rule_type == "and_implies":
            sample["antecedents"] = [normalize_literal(predicate) for predicate in rule["antecedents"]]
            sample["consequents"] = [normalize_literal(rule["consequent"])]
        elif rule_type == "or_implies":
            sample["antecedents"] = [normalize_literal(predicate) for predicate in rule["antecedents"]]
            sample["consequents"] = [normalize_literal(rule["consequent"])]
        elif rule_type == "xor":
            sample["atoms"] = [normalize_literal(predicate) for predicate in rule["atoms"]]
        else:
            continue
        normalized.append(sample)
    return normalized


def deduplicate_rules(rules: List[dict]) -> List[dict]:
    seen = set()
    deduplicated = []
    for rule in rules:
        key = json.dumps(
            {
                "rule_type": rule["rule_type"],
                "antecedents": rule["antecedents"],
                "consequents": rule["consequents"],
                "atoms": rule["atoms"],
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(rule)
    return deduplicated


def get_normalized_rules(constraints_path: str, rules_path: str, rule_source: str) -> List[dict]:
    if rule_source == "auto":
        rule_source = "hybrid" if os.path.exists(rules_path) else "legacy"

    rules = []
    if rule_source in {"legacy", "hybrid"}:
        rules.extend(normalize_legacy_rules(constraints_path))
    if rule_source in {"augmented", "hybrid"}:
        rules.extend(normalize_augmented_rules(rules_path))
    rules = deduplicate_rules(rules)
    if not rules:
        return normalize_legacy_rules(constraints_path)
    return rules


def rule_literals(rule: dict) -> List[dict]:
    return rule["antecedents"] + rule["consequents"] + rule["atoms"]


def ground_rules(rules: List[dict], facts: List[dict], templates: Dict[str, dict], uncountables: List[str]) -> List[dict]:
    fact_lookup: Dict[str, Dict[str, dict]] = {}
    for fact in facts:
        fact_lookup.setdefault(fact["subject"], {})[fact["predicate"]] = fact

    grounded = []
    for subject, subject_facts in fact_lookup.items():
        for rule in rules:
            grounded_rule = {
                "rule_id": rule["id"],
                "rule_type": rule["rule_type"],
                "source": rule["source"],
                "subject": subject,
                "antecedents": [],
                "consequents": [],
                "atoms": [],
            }

            for field in ("antecedents", "consequents", "atoms"):
                for literal in rule[field]:
                    relation, obj = literal["predicate"].split(",")
                    fact = subject_facts.get(literal["predicate"])
                    grounded_literal = {
                        "predicate": literal["predicate"],
                        "positive": literal["positive"],
                        "statement": Facts.implication2string(
                            templates=templates,
                            uncountables=uncountables,
                            subject=subject,
                            relation=relation,
                            symbol=True,
                            obj=obj,
                        ),
                        "belief": fact["belief"] if fact is not None else -1,
                    }
                    grounded_rule[field].append(grounded_literal)
            grounded.append(grounded_rule)
    return grounded


class Constraints(data.Dataset):

    def __init__(self, constraints_path:str, model_type:str):
        self.samples = Constraints.get_links(path=constraints_path)
        self.model_type = model_type

    def get_grounded_constraints(self, facts:dict, path:str) -> List[object]:
        """ Istantiate links into constraints for each subject (using a fact split)
            Returns:
                samples:List[object]    {antecedent, consequent, s_antecedent, s_consequent, g_antecedent, g_consequent} 
        """
        # Facts list -> hash table
        hash_facts = dict()
        for fact in facts: hash_facts.setdefault(fact["predicate"], dict())[fact["subject"]] = fact
        # NL formatting
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"), 
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"))
        # Constraints
        general_constraints = Constraints.get_links(path)
        samples = []
        for constraint in general_constraints: # for each principle
            knows_antecedent = constraint["antecedent"] in hash_facts.keys()
            knows_consequent = constraint["consequent"] in hash_facts.keys()
            # include grounded fact symbols if known in the training facts set 
            # add all constraints for which this antecedent is known
            if knows_antecedent:
                for subj, belief in hash_facts[constraint["antecedent"]].items():
                    rel, obj = constraint["consequent"].split(",") # extract the implied consequent
                    sample = {
                        "antecedent": belief["fact"], 
                        "neg_antecedent": belief["negated_fact"], 
                        "consequent": Facts.implication2string(templates=templates, uncountables=uncountables, subject=subj, relation=rel, symbol=True, obj=obj),
                        "neg_consequent": Facts.implication2string(templates=templates, uncountables=uncountables, subject=subj, relation=rel, symbol=False, obj=obj),
                        "s_antecedent": int(constraint["s_antecedent"]), 
                        "s_consequent": int(constraint["s_consequent"]),
                        "g_antecedent": belief["belief"],
                        "g_consequent": -1
                    }
                    # also the consequent is known
                    if knows_consequent and subj in hash_facts[constraint["consequent"]]: sample["g_consequent"] = hash_facts[constraint["consequent"]][subj]["belief"]
                    samples.append(sample)
            elif knows_consequent:
                for subj, belief in hash_facts[constraint["consequent"]].items():
                    rel, obj = constraint["antecedent"].split(",") # extract the implying antecedent
                    sample = {
                        "consequent": belief["fact"], 
                        "neg_consequent": belief["negated_fact"], 
                        "antecedent": Facts.implication2string(templates=templates, uncountables=uncountables, subject=subj, relation=rel, symbol=True, obj=obj),
                        "neg_antecedent": Facts.implication2string(templates=templates, uncountables=uncountables, subject=subj, relation=rel, symbol=False, obj=obj),
                        "s_antecedent": int(constraint["s_antecedent"]), 
                        "s_consequent": int(constraint["s_consequent"]),
                        "g_antecedent": -1,
                        "g_consequent": belief["belief"]
                    }
                    samples.append(sample)
        return samples

    @staticmethod
    def get_links(path) -> (List[object]):
        """ Load all logical constraints and index by source edge 
            Parameters:
                path:str            Path to the constraints file (json expected)
            Returns:
                links:List[obj]     List of {antecedent, consequent, symbol_ant, symbol_conseq}

        """
        with open(path) as f:
            constraints = json.load(f)
        links = []
        for rel in constraints["links"]:
            if rel["direction"] == "forward":
                source = rel["source"]
                source_symbol =  rel["weight"].split("_")[0] == "yes"
                target = rel["target"]
                target_symbol =  rel["weight"].split("_")[1] == "yes"
                sample = {"antecedent": source, "consequent": target, "s_antecedent": source_symbol, "s_consequent": target_symbol}
                links.append(sample)
        return links  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int) -> dict:
        return self.samples[idx]


class Facts():

    def __init__(self, facts_path, constraints, model_type):
        self.constraints = constraints
        self.facts_path = facts_path
        self.model_type = model_type

    def get_whole_set(self) -> dict:
        """  Convert the BeliefBank facts into a set of NL samples """
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"), 
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"))
        with open(self.facts_path) as f:
            facts = json.load(f) 
            samples = []
            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=True, obj=obj)
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact, "belief": int(belief == "yes")}
                    samples.append(sample)
        return samples
    
    def get_multihop_splits(self) -> dict:
        """  Convert the BeliefBank facts into a set of NL samples """
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"), 
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"))
        with open(self.facts_path) as f:
            facts = json.load(f) 
            c_antecedents = [c["antecedent"] for c in self.constraints]
            c_consequents = [c["consequent"] for c in self.constraints] 
            train_facts = []
            test_facts = []
            for subject, subject_facts in facts.items(): # all subject facts
                for key, belief in subject_facts.items(): # key is the relashionship, belief is the binary assignment
                    relation, obj = key.split(",")
                    # construct constraint sample
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=True, obj=obj)
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact, "belief": int(belief == "yes")}
                    # key is B
                    # First hop: A -> B, B is in consequents
                    # Second hop: B -> C, B is in antecedents
                    if key in c_consequents and key in c_antecedents: test_facts.append(sample)
                    # B is just an antecedent XOR a consequent (0,0) + (0,1) + (1,0)
                    else: train_facts.append(sample)

        return {"train": train_facts, "test": test_facts}
        
    def get_splits(self) -> dict:
        """  Convert the BeliefBank facts into a set of NL samples """
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"), 
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"))
        with open(self.facts_path) as f:
            facts = json.load(f) 
            c_antecedents = [c["antecedent"] for c in self.constraints]
            c_consequents = [c["consequent"] for c in self.constraints] 
            ant_facts = []
            con_facts = []
            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=True, obj=obj)
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact, "belief": int(belief == "yes")}
                    # test = consequents
                    if key in c_consequents and key not in c_antecedents: con_facts.append(sample)
                    # test = antecedents
                    else: ant_facts.append(sample) 
        idx_val = random.sample(range(0, len(ant_facts)-1), int(0.1 * len(ant_facts)))
        train_ant_facts = [ant_facts[idx] for idx in range(len(ant_facts)) if idx not in idx_val]
        val_ant_facts = [ant_facts[idx] for idx in idx_val]
        return {"train": train_ant_facts, "test": con_facts, "val": val_ant_facts}
        
    @staticmethod
    def implication2string(templates, uncountables, subject, relation, symbol, obj):
        """ From implication object, convert into natural relation"""
        this_template = templates[relation]
        X = Facts.noun_fluenterer(subject, uncountables)
        Y = Facts.noun_fluenterer(obj, uncountables, relation)
        rng = random.randint(0, 1)
        # questions: templates, templates_negated
        # statements: assertion_positive, assertion_negative
        if symbol: nl_question = this_template['assertion_positive'].format(X=X, Y=Y)
        else: nl_question = this_template['assertion_negative'].format(X=X, Y=Y)
        return nl_question
    
    @staticmethod
    def get_language_templates(templates_path, uncountables_path):
        """ Read helper structures from file"""
        with open(templates_path) as f:
            natural_relations = json.load(f) 
            f.close()
        with open(uncountables_path) as f:
            uncountables = f.read().split('\n')
            f.close()
        return natural_relations, uncountables

    @staticmethod
    def noun_fluenterer(noun, uncountables, relation=None):

        """
        Make a noun phrase 'fluenter' (more fluent) before putting it in a
        template.  note we only a.) check if the noun is in a list of known
        non-countables or has a relation with a certain type, and b.) look at the
        first letter of the input to decide whether to put a or an.

        :param noun: the noun (phrase) -- subject or object -- to make more fluent
        :param uncountables: the list of uncountables to compare to
        :param relation: BeliefBank relation
        :return: a string with the prettified noun phrase
        """

        if noun in uncountables:
            return noun

        if relation is not None:
            if relation in ['CapableOf', 'MadeOf', 'HasProperty']:
                return noun

        if noun[0] in ['a', 'e', 'i', 'o', 'u']:
            return 'an ' + noun

        return 'a ' + noun
