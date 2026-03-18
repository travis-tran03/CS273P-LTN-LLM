
import os
import random
import time
from typing import List

import sklearn.metrics as metrics
import torch as th
import transformers
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset.beliefbank import *
from utils.constraints import ConstraintManager
from utils.eval import *
from utils.ltn import LTNRuleManager

def save_checkpoint(model, tokenizer, optimizer, filename, epoch, checkpoints_path):
    epoch_path = os.path.join(checkpoints_path, f"epoch-{epoch}")
    ckpt_path = os.path.join(epoch_path, filename)
    if not os.path.exists(epoch_path): os.mkdir(epoch_path)
    state = {
        'epoch': epoch+1, 'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    model.save_model(ckpt_path)
    tokenizer.save_model(ckpt_path)
    th.save(state, ckpt_path)

DEFAULT_CONFIG = {
    "runs": 1,
    "lr": 0.0003,
    "lmbda": 1,
    "use_table_truth": False,
    "factuality": True,
    "training": "combined",
    "staged_facts": True,
    "staged_constraints": True,
    "constraint_type": "all",
    "logic_backend": "sdd",
    "rule_source": "legacy",
    "logic_weight": 1.0,
    "factual_weight": 1.0,
    "compute_perplexity": False,
    "implication_floor": 0.5,
    "rule_batch_size": 32,
    "sat_agg_p": 2.0,
    "prob_chunk_size": 4,
    "max_train_rules": -1,
}


def collate_rule_batch(batch):
    return batch

class Trainer():

    def __init__(
            self, model, 
            constraint_mg,
            lr,
            wandb,
            optimizer,
            checkpoints_path,
            gpu_id,
            config,
            run_parallel,
            val_interval = 1,
        ):
        
        self.lr = lr
        self.model = model
        self.constraint_mg = constraint_mg
        self.wandb = wandb
        self.logging = config["wandb"]
        self.val_interval = val_interval
        self.checkpoints_path = checkpoints_path
        self.optimizer = optimizer

        self.config = config
        self.num_epochs = config["epochs"]
        self.lmbda = config["lmbda"] if "lmbda" in config else DEFAULT_CONFIG["lmbda"]
        self.accumulation_steps = config["accumulation_steps"]
        self.factuality = config["factuality"] if "factuality" in config else DEFAULT_CONFIG["factuality"]
        self.batch_size = config["batch_size"] // config["accumulation_steps"]
        self.use_table_truth = config["use_table_truth"] if "use_table_truth" in config else DEFAULT_CONFIG["use_table_truth"]
        self.constraint_type = config["constraint_type"] if "constraint_type" in config else DEFAULT_CONFIG["constraint_type"]
        self.logic_backend = config["logic_backend"] if "logic_backend" in config else DEFAULT_CONFIG["logic_backend"]
        self.rule_source = config["rule_source"] if "rule_source" in config else DEFAULT_CONFIG["rule_source"]
        self.logic_weight = config["logic_weight"] if "logic_weight" in config else DEFAULT_CONFIG["logic_weight"]
        self.factual_weight = config["factual_weight"] if "factual_weight" in config else DEFAULT_CONFIG["factual_weight"]
        self.compute_perplexity = config["compute_perplexity"] if "compute_perplexity" in config else DEFAULT_CONFIG["compute_perplexity"]
        self.implication_floor = config["implication_floor"] if "implication_floor" in config else DEFAULT_CONFIG["implication_floor"]
        self.rule_batch_size = config.get("rule_batch_size", DEFAULT_CONFIG["rule_batch_size"])
        self.sat_agg_p = config.get("sat_agg_p", DEFAULT_CONFIG["sat_agg_p"])
        self.prob_chunk_size = config.get("prob_chunk_size", DEFAULT_CONFIG["prob_chunk_size"])
        self.max_train_rules = config.get("max_train_rules", DEFAULT_CONFIG["max_train_rules"])
        self.sdd_weight = config.get("sdd_weight", 1.0)
        self.ltn_weight = config.get("ltn_weight", 1.0)
        self.run_parallel = run_parallel
        self.rule_manager = LTNRuleManager(
            implication_floor=self.implication_floor,
            p=self.sat_agg_p,
        )
        
        if config["lr_scheduler"]: 
            print("[-] Using lr scheduler: CosineAnnealingLR")
            self.LRscheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.num_epochs)
        else: self.LRscheduler = None

        self.gpu_id = gpu_id
        if self.run_parallel: 
            self.model = DDP(model, device_ids=[gpu_id])
            self.model_net = self.model.module
        else: 
            self.model = model
            self.model_net = model

        self.templates, self.uncountables = Facts.get_language_templates(
            templates_path=os.path.join("data", "beliefbank", "templates.json"), 
            uncountables_path=os.path.join("data", "beliefbank", "non_countable.txt"))
        
        print(f"[-] Model: {self.model_net.model_hf_name}")
        print(f"[-] Learning rate: {self.lr}")
        print(f"[-] Batch size: {self.batch_size}")
        print(f"[-] Accumulation steps: {self.accumulation_steps}")
        
        # Setting filenames
        backend_name = self.logic_backend if self.logic_backend in ("ltn", "hybrid") else self.constraint_type
        self.ckpt_name = f"{backend_name}_{self.model_net.model_hf_name.split('/')[1]}"
        print(f"[!] Setting checkpoint save path to: {self.ckpt_name}")

        self.path_outputs_log = os.path.join(f"outputs_{time.strftime('%Y%m%d-%H%M%S')}_{self.model_net.model_hf_name.split('/')[1]}.log")
        print(f"[!] Setting outputs log to: {self.path_outputs_log}")

    def save_model(self, path):
        if not os.path.exists(path): os.mkdir(path)
        self.model_net.model.save_pretrained(path)
        self.model_net.tokenizer.save_pretrained(path)

    def is_verbose(self):
        return "verbose" in self.config and self.config["verbose"]

    def log_outputs(self, line):
        with open(self.path_outputs_log, "a") as f:
            f.write(line)

    def log_step(self, log:object, progressbar:object=None):
        """ Log scores to wandb and return log string """
        if self.logging and (self.gpu_id == 0 or not self.run_parallel): self.wandb.log(log)
        timing = ""
        metrics = ""
        for key, val in log.items():
            if key == "batch_s":
                timing = f"[{val:.1f}s] "
            else:
                metrics += f"{key}: {val:.2f}; "
        printout = timing + metrics
        if progressbar is not None:
            progressbar.update(1)
            tqdm.write(printout)
        else: print(printout)

    def run_eval(self):
        print(f"[+] Evaluating model.")
        data = self.prepare_data()
        eval_prompts = self.config.get("eval_prompts", [0, 1])
        # Test
        if(self.gpu_id == 0 or not self.run_parallel):
            for prompt_idx in eval_prompts:
                self.score(data=data, mode="test", split="calibration", prompt_idx=prompt_idx)
                self.score(data=data, mode="test", split="silver", prompt_idx=prompt_idx)

            if self.compute_perplexity:
                print(f"\n[-] Scoring perplexity...")
                test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                ppl = self.model_net.get_perplexity(data=test["text"], window_size=4096)
                print(f"\t Perplexity: {ppl}")

    def run_train(self, mode="combined"):
        print(f"[+] Training in {mode} mode.")
        data = self.prepare_data()
        # Train
        for e in range(self.num_epochs):
            # Validation
            if (e % self.val_interval == 0) and (self.gpu_id == 0 or not self.run_parallel):
                print(f"[-] Evaluation @ epoch {e}...")
                self.epoch_eval(data=data, epoch=e)  
            if self.run_parallel: th.distributed.barrier()

            print("\n[-] Training loop...")
            if self.logic_backend == "ltn":
                self.epoch_ltn(rules=data["rules"]["train"], facts=data["facts"]["train"]["calibration"], epoch=e)
            elif self.logic_backend == "hybrid":
                self.epoch_hybrid(
                    constraints=data["constraints"]["train"],
                    rules=data["rules"]["train"],
                    facts=data["facts"]["train"]["calibration"],
                    epoch=e,
                )
            else:
                self.epoch(constraints=data["constraints"]["train"], epoch=e)
            if self.LRscheduler is not None: self.LRscheduler.step()

            # Saving each epoch
            # save_checkpoint(
            #     epoch=e, 
            #     model=self.model_net.model, 
            #     tokenizer=self.model_net.tokenizer,
            #     optimizer=self.optimizer,
            #     checkpoints_path=self.checkpoints_path, 
            #     filename=self.ckpt_name
            # )
            save_path = os.path.join(self.checkpoints_path, f"epoch-{e}-{self.ckpt_name}")
            self.save_model(save_path)

        # Test
        print("\n[-] Final test of the model...")
        if(self.gpu_id == 0 or not self.run_parallel):
            self.score(data=data, mode="test", split="calibration", prompt_idx=0)
            self.score(data=data, mode="test", split="silver", prompt_idx=0)
            self.score(data=data, mode="test", split="calibration", prompt_idx=1)
            self.score(data=data, mode="test", split="silver", prompt_idx=1)   
        if self.run_parallel: th.distributed.barrier()

        print("\n[-] Saving checkpoint")
        # save_checkpoint(
        #     epoch=self.num_epochs, 
        #     model=self.model_net.model, 
        #     tokenizer=self.model_net.tokenizer,
        #     optimizer=self.optimizer,
        #     checkpoints_path=self.checkpoints_path, 
        #     filename=self.ckpt_name
        # )
        save_path = os.path.join(self.checkpoints_path, f"epoch-{e}-{self.ckpt_name}")
        self.save_model(save_path)
        print(f"[-] Checkpoint correctly stored in: {save_path}")

    def epoch_eval(self, data:object, epoch:int):
        """ 
            Log validation scores and return early stop signal 
        """
        self.score(data=data, mode="val", split="calibration", prompt_idx=0)
        self.score(data=data, mode="val", split="calibration", prompt_idx=1)

    def epoch_facts(self, dataset, epoch=None):
        """ 
            One epoch training on facts 
        """ 
        e_loss = []
        progressbar = tqdm(dataset)
        batch_idx = 0
        for batch in progressbar:
            loss = self.model_net.get_facts_loss(statements=batch["fact"], labels=batch["belief"].tolist())
            loss = loss / self.accumulation_steps
            loss.backward()
            # Gradient accumulation
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(dataset)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            e_loss.append(loss.item())
            log = {
                "train/factual_loss":  loss.item(),
                "epoch": epoch
            }
            self.log_step(log=log, progressbar=progressbar)
            batch_idx += 1
        return th.mean(th.tensor(e_loss))

    def epoch(self, constraints, epoch=None):
        """ 
            Combined training epoch 
        """
        e_loss = []
        batch_idx = 0
        progressbar = tqdm((constraints))
        for batch in progressbar:

            prompt_idx = random.choice([0, 1]) # randomize prompt at each fwd pass
            # Preprocess
            if self.model_net.is_decoder():
                premises = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["antecedent"]]      # B, 1
                hypothesis = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["consequent"]]    # B, 1
                neg_premises = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["neg_antecedent"]]      # B, 1
                neg_hypothesis = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["neg_consequent"]]    # B, 1
            
            elif self.model_net.is_seq2seq():
                premises = [p for p in batch["antecedent"]]      # B, 1
                hypothesis = [p for p in batch["consequent"]]    # B, 1
                neg_premises = [p for p in batch["neg_antecedent"]]      # B, 1
                neg_hypothesis = [p for p in batch["neg_consequent"]]    # B, 1
            
            # Model Inference
            # groundings, symbols and inferred beliefs vary depending on the constraint
            # so the losses change: negation constraint implies applying on both premise 
            # and hypothesis (one at a time)
            log = { "epoch": epoch }
            p1, p2 = self.model_net.get_formula_beliefs(s1=premises, s2=hypothesis, label=FORMATS[prompt_idx]["label"])  
            
            # Preparing inputs
            if self.constraint_type == "negation":
                ground_facts = None
                constraint_symbols = th.ones((batch["s_antecedent"].shape[0], 2)).to(self.gpu_id)
            else:
                ground_facts = list(zip(batch["g_antecedent"].tolist(), batch["g_consequent"].tolist())) # B, 2
                constraint_symbols = th.stack((batch["s_antecedent"], batch["s_consequent"]), dim=1) # B, 2

            # Supporting all constraints combined
            if ConstraintManager.need_negations(self.constraint_type):
                p1_not, p2_not = self.model_net.get_formula_beliefs(s1=neg_premises, s2=neg_hypothesis, label=FORMATS[prompt_idx]["label"])
            else: p1_not, p2_not = None, None

            loss = self.constraint_mg.sl(
                p1=p1, p2=p2, p1_not=p1_not, p2_not=p2_not,
                batch_symbols=constraint_symbols, 
                batch_facts=ground_facts,
                constraint=self.constraint_type
            )

            # Backprop
            loss = loss / self.accumulation_steps
            loss.backward()
            log["train/loss"] = loss.item()
            
            # Gradient accumulation
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(constraints)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.log_step(log=log, progressbar=progressbar)
            e_loss.append(loss.item())
            batch_idx += 1

        return th.mean(th.tensor(e_loss))

    def epoch_ltn(self, rules, facts, epoch=None):
        """
            LTN-style training with factual supervision plus fuzzy rule satisfaction.
        """
        e_loss = []
        batch_idx = 0
        progressbar = tqdm(rules)
        facts_iterator = iter(facts)
        for batch in progressbar:
            batch_start = time.time()
            log = {"epoch": epoch}
            try:
                fact_batch = next(facts_iterator)
            except StopIteration:
                facts_iterator = iter(facts)
                fact_batch = next(facts_iterator)

            unique_statements = []
            seen = set()
            for sample in batch:
                for literal in sample["antecedents"] + sample["consequents"] + sample["atoms"]:
                    statement = literal["statement"]
                    if statement in seen:
                        continue
                    seen.add(statement)
                    unique_statements.append(statement)

            if batch and unique_statements:
                prob_chunk_size = self.prob_chunk_size
                all_probs = []
                for i in range(0, len(unique_statements), prob_chunk_size):
                    chunk = unique_statements[i : i + prob_chunk_size]
                    all_probs.append(self.model_net.get_fact_probabilities(chunk))
                statement_probs = th.cat(all_probs, dim=0)
                statement_truths = {statement: statement_probs[idx] for idx, statement in enumerate(unique_statements)}
                truths = self.rule_manager.batch_rule_truths(
                    batch=batch, statement_truths=statement_truths, filter_applicable=False,
                )
                logic_loss = self.rule_manager.logic_loss(truths).to(self.gpu_id)
                log["train/rule_satisfaction"] = truths.mean().item()
                log["train/active_rules"] = float(truths.numel())
            else:
                logic_loss = th.tensor(0.0, device=self.gpu_id)
                log["train/rule_satisfaction"] = 0.0
                log["train/active_rules"] = 0.0

            if self.factual_weight > 0:
                factual_loss = self.model_net.get_facts_loss(
                    statements=fact_batch["fact"],
                    labels=fact_batch["belief"].tolist(),
                )
            else:
                factual_loss = th.tensor(0.0, device=self.gpu_id)

            loss = (self.logic_weight * logic_loss) + (self.factual_weight * factual_loss)
            loss = loss / self.accumulation_steps
            loss.backward()

            batch_time = time.time() - batch_start
            log["train/ltn_loss"] = logic_loss.item()
            log["train/factual_loss"] = factual_loss.item()
            log["train/loss"] = loss.item()
            log["batch_s"] = batch_time

            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(rules)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.log_step(log=log, progressbar=progressbar)
            e_loss.append(loss.item())
            batch_idx += 1

        return th.mean(th.tensor(e_loss))

    def epoch_hybrid(self, constraints, rules, facts, epoch=None):
        """
            Hybrid training: SDD semantic loss for binary constraints plus
            LTN fuzzy loss for complex rules (and_implies, or_implies, xor).
        """
        e_loss = []
        batch_idx = 0
        progressbar = tqdm(constraints)
        rules_iterator = iter(rules)
        facts_iterator = iter(facts)

        for batch in progressbar:
            batch_start = time.time()
            log = {"epoch": epoch}

            # --- SDD loss (binary constraints) ---
            prompt_idx = random.choice([0, 1])
            if self.model_net.is_decoder():
                premises = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["antecedent"]]
                hypothesis = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["consequent"]]
                neg_premises = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["neg_antecedent"]]
                neg_hypothesis = [FORMATS[prompt_idx]["prompt"].format(fact=p) for p in batch["neg_consequent"]]
            elif self.model_net.is_seq2seq():
                premises = list(batch["antecedent"])
                hypothesis = list(batch["consequent"])
                neg_premises = list(batch["neg_antecedent"])
                neg_hypothesis = list(batch["neg_consequent"])

            p1, p2 = self.model_net.get_formula_beliefs(
                s1=premises, s2=hypothesis, label=FORMATS[prompt_idx]["label"]
            )
            if self.constraint_type == "negation":
                ground_facts = None
                constraint_symbols = th.ones((batch["s_antecedent"].shape[0], 2)).to(self.gpu_id)
            else:
                ground_facts = list(zip(batch["g_antecedent"].tolist(), batch["g_consequent"].tolist()))
                constraint_symbols = th.stack((batch["s_antecedent"], batch["s_consequent"]), dim=1)

            if ConstraintManager.need_negations(self.constraint_type):
                p1_not, p2_not = self.model_net.get_formula_beliefs(
                    s1=neg_premises, s2=neg_hypothesis, label=FORMATS[prompt_idx]["label"]
                )
            else:
                p1_not, p2_not = None, None

            sdd_loss = self.constraint_mg.sl(
                p1=p1, p2=p2, p1_not=p1_not, p2_not=p2_not,
                batch_symbols=constraint_symbols,
                batch_facts=ground_facts,
                constraint=self.constraint_type,
            )

            # --- LTN loss (complex rules) ---
            try:
                rule_batch = next(rules_iterator)
            except StopIteration:
                rules_iterator = iter(rules)
                rule_batch = next(rules_iterator)

            unique_statements = []
            seen = set()
            for sample in rule_batch:
                for literal in sample["antecedents"] + sample["consequents"] + sample["atoms"]:
                    stmt = literal["statement"]
                    if stmt not in seen:
                        seen.add(stmt)
                        unique_statements.append(stmt)

            if rule_batch and unique_statements:
                prob_chunk_size = self.prob_chunk_size
                all_probs = []
                for i in range(0, len(unique_statements), prob_chunk_size):
                    chunk = unique_statements[i : i + prob_chunk_size]
                    all_probs.append(self.model_net.get_fact_probabilities(chunk))
                statement_probs = th.cat(all_probs, dim=0)
                statement_truths = {s: statement_probs[idx] for idx, s in enumerate(unique_statements)}
                truths = self.rule_manager.batch_rule_truths(
                    batch=rule_batch, statement_truths=statement_truths, filter_applicable=False,
                )
                ltn_loss = self.rule_manager.logic_loss(truths).to(self.gpu_id)
                log["train/rule_satisfaction"] = truths.mean().item()
                log["train/active_rules"] = float(truths.numel())
            else:
                ltn_loss = th.tensor(0.0, device=self.gpu_id)
                log["train/rule_satisfaction"] = 0.0
                log["train/active_rules"] = 0.0

            # --- Factual loss ---
            try:
                fact_batch = next(facts_iterator)
            except StopIteration:
                facts_iterator = iter(facts)
                fact_batch = next(facts_iterator)

            if self.factual_weight > 0:
                factual_loss = self.model_net.get_facts_loss(
                    statements=fact_batch["fact"],
                    labels=fact_batch["belief"].tolist(),
                )
            else:
                factual_loss = th.tensor(0.0, device=self.gpu_id)

            # --- Combined loss ---
            loss = (
                self.sdd_weight * sdd_loss
                + self.ltn_weight * ltn_loss
                + self.factual_weight * factual_loss
            )
            loss = loss / self.accumulation_steps
            loss.backward()

            batch_time = time.time() - batch_start
            log["train/sdd_loss"] = sdd_loss.item()
            log["train/ltn_loss"] = ltn_loss.item()
            log["train/factual_loss"] = factual_loss.item()
            log["train/loss"] = loss.item()
            log["batch_s"] = batch_time

            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(constraints)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.log_step(log=log, progressbar=progressbar)
            e_loss.append(loss.item())
            batch_idx += 1

        return th.mean(th.tensor(e_loss))

    def get_beliefs_facts(self, facts, prompt_idx=0):
        """ Get model's beliefs """
        print(f"\n[-] Querying facts for prompt {prompt_idx}...")

        model_beliefs = dict()
        ref_beliefs = dict()

        answ_model_beliefs = []
        answ_model_negated_beliefs = []
        answ_ground_beliefs = []

        trues = []

        if self.is_verbose(): iterator = tqdm(facts)
        else: iterator = facts

        for batch in iterator:
            with th.no_grad():

                # Pre-processing
                facts = preprocess_batch(batch=batch["fact"], prompt_format=prompt_idx)
                negated_facts = preprocess_batch(batch=batch["negated_fact"], prompt_format=prompt_idx)

                # Query
                answers = self.model_net.infer_fact(facts)
                answers_negations = self.model_net.infer_fact(negated_facts)

                # Post-processing
                raw_statements, statements = postprocess_answers(prompts=facts, answers=answers, prompt_format=prompt_idx)
                raw_neg_statements, neg_statements = postprocess_answers(prompts=negated_facts, answers=answers_negations, prompt_format=prompt_idx)
                
                # Printing outputs 
                for l in list(zip(facts, raw_statements, statements, negated_facts, raw_neg_statements, neg_statements)): self.log_outputs(line=f"{l}\n")

                # Creating a dictionary of {key: property, value: statement}
                batch_beliefs = batch["belief"].tolist()
                batch_predicates = batch["predicate"]
                for idx, subj in enumerate(batch["subject"]):
                    model_beliefs.setdefault(subj, dict())[batch_predicates[idx]] = statements[idx]
                    ref_beliefs.setdefault(subj, dict())[batch_predicates[idx]] = batch_beliefs[idx]

                # Storing beliefs
                trues += statements
                answ_model_beliefs += statements
                answ_model_negated_beliefs += neg_statements
                answ_ground_beliefs += batch_beliefs
        
        print(f"\n[-] Distribution:")
        print(f"# \"{FORMATS[prompt_idx]['label']}\": {sum(trues)}\n# total: {len(trues)}")

        return {
            "beliefs": answ_model_beliefs, 
            "negated_beliefs": answ_model_negated_beliefs, 
            "ground_beliefs": answ_ground_beliefs, 
            "dict_beliefs": model_beliefs, 
            "ref_beliefs": ref_beliefs
        }

    def score_facts(self, facts) -> dict:
        """
            Test model's beliefs against a knowledge base
        """
        f1 = metrics.f1_score(facts["ground_beliefs"], facts["beliefs"])
        negation_consistency = Metrics.negation_consistency(facts["beliefs"], facts["negated_beliefs"])
        return {"f1": f1, "negation_consistency": negation_consistency}

    def score_logic_sdd(self, facts, constraints) -> dict:
        """
            Test model's beliefs against logical constraints
        """
        self_consistency = Metrics.consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["dict_beliefs"], constraints=constraints)
        self_inverse_consistency = Metrics.inverse_consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["dict_beliefs"], constraints=constraints)
        self_multihop_consistency = Metrics.multihop_consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["dict_beliefs"], constraints=constraints)
        
        consistency = Metrics.consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["ref_beliefs"], constraints=constraints)
        inverse_consistency = Metrics.inverse_consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["ref_beliefs"], constraints=constraints)
        multihop_consistency = Metrics.multihop_consistency(model_beliefs=facts["dict_beliefs"], ref_beliefs=facts["ref_beliefs"], constraints=constraints)
        
        satisfiability = Metrics.satisfiability(model_beliefs=facts["dict_beliefs"], constraints=constraints)

        return {
            "self_consistency": self_consistency, 
            "self_inverse_consistency": self_inverse_consistency, 
            "self_multihop_consistency": self_multihop_consistency,
            "satisfiability": satisfiability,
            "consistency": consistency, 
            "inverse_consistency": inverse_consistency, 
            "multihop_consistency": multihop_consistency
        }

    def score_logic_ltn(self, facts, rules) -> dict:
        avg_rule_satisfaction = self.rule_manager.average_satisfaction(
            grounded_rules=rules,
            belief_lookup=facts["dict_beliefs"],
        )
        return {
            "avg_rule_satisfaction": avg_rule_satisfaction,
        }

    def score(self, data:dict, mode:str, split:str, prompt_idx:int=DEFAULT_PROMPT_FORMAT): 
        """
            Overall model benchmark (logic + facts)
        """
        print(f"\n[-] Computing scores on {split} split in {mode} mode.")
        print(f"[-] Using prompt format: {prompt_idx}")

        beliefs = self.get_beliefs_facts(facts=data["facts"][mode][split], prompt_idx=prompt_idx)
        
        print(f"\n[{mode}-{split}] Scoring factuality...")
        factuality = self.score_facts(facts=beliefs)

        # Unified eval: always compute LTN rule satisfaction as common metric
        print(f"\n[{mode}-{split}] Scoring LTN rule satisfaction...")
        ltn_logic = self.score_logic_ltn(facts=beliefs, rules=data["rules"][mode][split])

        # Also compute SDD metrics when constraint data is available
        sdd_logic = {}
        if self.logic_backend in ("sdd", "hybrid"):
            print(f"\n[{mode}-{split}] Scoring SDD constraints...")
            sdd_logic = self.score_logic_sdd(facts=beliefs, constraints=data["constraints"][mode])

        scores = {}
        for k, v in factuality.items(): scores[f"{mode}/prompt{prompt_idx}_{split}_{k}"] = v
        for k, v in ltn_logic.items(): scores[f"{mode}/prompt{prompt_idx}_{split}_{k}"] = v
        for k, v in sdd_logic.items(): scores[f"{mode}/prompt{prompt_idx}_{split}_{k}"] = v

        # Compute perplexity only during evaluation
        if self.compute_perplexity and mode == "val" and prompt_idx == DEFAULT_PROMPT_FORMAT:
            print(f"\n[{mode}-{split}] Scoring perplexity...")
            test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            ppl = self.model_net.get_perplexity(data=test["text"], window_size=4096)
            scores[f"{mode}/perplexity"] = ppl

        # Compute ants/cons F1 only at test time
        if mode == "test":
            print(f"\n [{mode}-{split}] Scoring antecedents-consequents")
            beliefs_ants = self.get_beliefs_facts(facts=data["facts"][mode][f"{split}_antecedents"], prompt_idx=prompt_idx)
            antecedents_f1 = self.score_facts(facts=beliefs_ants)["f1"]
            beliefs_cons = self.get_beliefs_facts(facts=data["facts"][mode][f"{split}_consequents"], prompt_idx=prompt_idx)
            consequents_f1 = self.score_facts(facts=beliefs_cons)["f1"]
            scores[f"{mode}/prompt{prompt_idx}_{split}_antecedents_f1"] = antecedents_f1
            scores[f"{mode}/prompt{prompt_idx}_{split}_consequents_f1"] = consequents_f1
        
        # Log all scores
        self.log_step(log=scores)
        return scores

    def prepare_data(self):
        """
            Preparing data distributions and splits
        """
        # Training
        data = get_dataset(model_type=self.model_net.get_model_type(), config=self.config)
        train_facts = DataLoader(data["facts"]["calibration"]["train"], batch_size=self.batch_size, sampler=(DistributedSampler(data["facts"]["calibration"]["train"]) if self.run_parallel else None), shuffle=(not self.run_parallel))
        train_constraints = DataLoader(data["constraints"]["train"], batch_size=self.batch_size, sampler=(DistributedSampler(data["constraints"]["train"]) if self.run_parallel else None), shuffle=(not self.run_parallel))
        
        # Subsample eval facts if max_eval_facts is set
        max_eval_facts = self.config.get("max_eval_facts", -1)
        def _maybe_subsample(dataset, label):
            if max_eval_facts > 0 and len(dataset) > max_eval_facts:
                indices = random.sample(range(len(dataset)), max_eval_facts)
                subset = Subset(dataset, indices)
                print(f"[!] Subsampled {label}: {max_eval_facts} / {len(dataset)}")
                return subset
            return dataset

        # Eval / test splits
        eval_calibration = _maybe_subsample(data["facts"]["calibration"]["complete"], "calibration eval facts")
        eval_silver = _maybe_subsample(data["facts"]["silver"]["complete"], "silver eval facts")
        test_all_calibration_facts = DataLoader(eval_calibration, batch_size=self.batch_size, shuffle=False)
        test_all_silver_facts = DataLoader(eval_silver, batch_size=self.batch_size, shuffle=False)

        # Validation
        val_calibration_facts = DataLoader(data["facts"]["calibration"]["val"], batch_size=self.batch_size, shuffle=False)

        # Testing
        all_constraints = DataLoader(data["constraints"]["all"], batch_size=self.batch_size, shuffle=False)
        test_calibration_facts = DataLoader(
            _maybe_subsample(data["facts"]["calibration"]["test"], "calibration test facts"),
            batch_size=self.batch_size, shuffle=False,
        )
        test_silver_facts = DataLoader(
            _maybe_subsample(data["facts"]["silver"]["test"], "silver test facts"),
            batch_size=self.batch_size, shuffle=False,
        )
        train_calibration_facts = DataLoader(
            _maybe_subsample(data["facts"]["calibration"]["train"], "calibration train facts"),
            batch_size=self.batch_size, shuffle=False,
        )
        train_silver_facts = DataLoader(
            _maybe_subsample(data["facts"]["silver"]["train"], "silver train facts"),
            batch_size=self.batch_size, shuffle=False,
        )

        payload = {
            "constraints": {
                "train": train_constraints,
                "val": all_constraints,
                "test": all_constraints,
            },
            "facts": {
                "train": {
                    "calibration": train_facts
                },
                "val": {
                    "calibration": val_calibration_facts,
                },
                "test": {
                    "calibration": test_all_calibration_facts,
                    "silver": test_all_silver_facts,
                    "calibration_consequents": test_calibration_facts,
                    "silver_consequents": test_silver_facts,
                    "calibration_antecedents": train_calibration_facts,
                    "silver_antecedents": train_silver_facts
                },
            }
        }

        # Always load rules for unified eval; subset training rules for ltn/hybrid
        train_rules = data["rules"]["train"]
        if self.logic_backend in ("ltn", "hybrid") and self.max_train_rules > 0 and len(train_rules) > self.max_train_rules:
            indices = random.sample(range(len(train_rules)), self.max_train_rules)
            train_rules = Subset(train_rules, indices)
            print(f"[!] Subsampled training rules: {len(train_rules)} / {len(data['rules']['train'])}")
        payload["rules"] = {
            "train": DataLoader(
                train_rules,
                batch_size=self.rule_batch_size,
                sampler=(DistributedSampler(train_rules) if self.run_parallel else None),
                shuffle=(not self.run_parallel),
                collate_fn=collate_rule_batch,
            ),
            "val": {
                "calibration": data["rules"]["val"]["calibration"],
            },
            "test": {
                "calibration": data["rules"]["test"]["calibration"],
                "silver": data["rules"]["test"]["silver"],
            },
        }

        return payload