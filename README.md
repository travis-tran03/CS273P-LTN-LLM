# CS273P-LTN-LLM

Improving Logical Consistency in Large Language Models via Logic Tensor Networks

**Authors:** Anishwar Chakraborty, Travis Tran  
**Course:** CS 273P, UC Irvine  
**Based on:** [Logically Consistent Language Models via Neuro-Symbolic Integration](https://arxiv.org/abs/2409.13724) (Calanzone et al., ICLR 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2409.13724-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2409.13724)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Overview

Large language models can assign plausible truth values to individual statements but frequently contradict themselves when reasoning across related facts. This project investigates whether **Logic Tensor Networks (LTNs)** can improve logical consistency in LLM outputs while preserving factual accuracy.

We treat a pretrained LLM (LLaMA 3.2 1B) as a soft truth estimator and feed its output probabilities into an LTN framework. The LTN evaluates differentiable logical constraints (universal, conjunction, disjunction, equivalence, exclusive-or) and produces a logic loss that is back-propagated into the LLM alongside a standard factual supervision loss.

Four configurations are compared:
1. **Base** - the pretrained LLM without any logic training
2. **SDD** - neuro-symbolic training using Sentential Decision Diagrams for propositional semantic loss
3. **LTN** - training with Logic Tensor Networks using fuzzy first-order logic
4. **Hybrid** - a combined objective merging both SDD and LTN losses

## Repository Structure

```
CS273P-LTN-LLM/
├── configs/                # JSON configuration files for training and evaluation
│   ├── train_llama32_1b_sdd.json
│   ├── train_llama32_1b_ltn.json
│   ├── train_llama32_1b_hybrid.json
│   ├── eval_llama32_1b_base.json
│   ├── eval_llama32_1b_sdd.json
│   ├── eval_llama32_1b_ltn.json
│   └── eval_llama32_1b_hybrid.json
├── data/
│   └── beliefbank/         # BeliefBank dataset (downloaded during setup)
├── dataset/
│   ├── beliefbank.py       # BeliefBank data loader and augmented-rule handling
│   └── dataset.py          # Base dataset utilities
├── models/
│   └── loco/
│       ├── model.py        # QA model wrapper with LoRA and 4-bit quantization
│       ├── trainer.py      # Training loop (SDD, LTN, hybrid modes)
│       └── prompts.py      # Prompt templates and answer post-processing
├── utils/
│   ├── ltn.py              # LTN fuzzy connectives and rule evaluation
│   ├── constraints.py      # SDD constraint manager
│   ├── eval.py             # Evaluation metrics (consistency, satisfiability)
│   └── training.py         # Training utilities
├── scripts/
│   ├── setup.sh            # Automated environment and data setup
│   ├── generate_rules_llm.py  # LLM-based rule augmentation pipeline
│   ├── train_llama.sh      # Example training launch script
│   └── eval_llama.sh       # Example evaluation launch script
├── results/                # JSON evaluation outputs
├── checkpoints/            # Saved LoRA adapters (created during training)
├── report/
│   └── report.tex          # LaTeX project report
├── run.py                  # Main entry point for training and evaluation
├── requirements.txt        # Python dependencies
├── environment.yml         # Full conda environment specification
└── LTN_LLM_training_for_first_order_logic.ipynb  # Jupyter notebook walkthrough
```

## Prerequisites

- **OS:** Linux (recommended) or Windows with WSL
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA support (required). At least 8 GB VRAM recommended for 4-bit quantized LLaMA 3.2 1B.
- **CUDA:** 11.8+ (or a compatible version for your PyTorch install)
- **Hugging Face account:** Access to `meta-llama/Llama-3.2-1B` requires accepting the LLaMA license on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B)

## Installation

### Option A: Automated Setup (Linux / WSL)

```bash
git clone https://github.com/travis-tran03/CS273P-LTN-LLM.git
cd CS273P-LTN-LLM
bash scripts/setup.sh
```

This script will:
1. Create the `checkpoints/` and `data/` directories
2. Download the BeliefBank dataset via `gdown`
3. Install the conda environment from `environment.yml`

After setup, activate the environment:

```bash
conda activate loco_lms
```

### Option B: Manual Setup (pip)

```bash
git clone https://github.com/travis-tran03/CS273P-LTN-LLM.git
cd CS273P-LTN-LLM
```

**1. Create and activate a virtual environment:**

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

**2. Install PyTorch with CUDA support:**

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select the install command matching your CUDA version. For example, with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Install project dependencies:**

```bash
pip install -r requirements.txt
```

**4. Download the BeliefBank dataset:**

```bash
mkdir -p checkpoints data
pip install gdown
gdown 1uAf4VEbok7CI5W34F7SzsJGU2t3YzN5W -O data/datasets.tar.gz
cd data && tar -xzvf datasets.tar.gz && cd ..
```

After extraction you should have the following files inside `data/beliefbank/`:
- `silver_facts.json` (12,363 labelled facts for 85 entities)
- `constraints_v2.json` (4,058 implication/negation constraint edges)
- `calibration_facts.json` (calibration split)
- `relation_vocabulary.txt`

**5. Authenticate with Hugging Face:**

```bash
huggingface-cli login
```

Enter your access token when prompted. This is required to download the gated LLaMA 3.2 1B model weights on first run.

## Data Augmentation (Optional)

The project includes a rule-augmentation pipeline that uses a Gemini LLM to generate richer logical rules beyond the original implication/negation constraints in BeliefBank.

**1. Create a `.env` file** in the repository root:

```
GEMINI_API_KEY=your_api_key_here
```

**2. Run the rule generator:**

```bash
python scripts/generate_rules_llm.py --model gemini-2.5-flash-lite
```

This produces `data/beliefbank/beliefbank_augmented.json` containing five new rule types:
- **XOR** - at most one relation in a set can be true per entity
- **AND-implies** - a conjunction of relations implies another
- **OR-implies** - at least one relation in a set implies a consequent
- **Equivalence** - two relations are true/false together
- **Universal** - if relation A holds, relation B must also hold

Each candidate rule is validated against the silver facts and kept only if it holds for all 85 entities.

If you skip this step, training still works using the original BeliefBank constraints. Set `"rule_source": "legacy"` in your config to use only the original constraints, or `"rule_source": "hybrid"` to use both original and augmented rules (requires the augmented file).

## Configuration

All training and evaluation runs are driven by JSON config files in `configs/`. Key fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Hugging Face model ID (e.g. `meta-llama/Llama-3.2-1B`) |
| `task` | string | `"train"` or `"eval"` |
| `training` | string | Training mode, typically `"combined"` (factual + logic) |
| `logic_backend` | string | `"sdd"`, `"ltn"`, or `"hybrid"` |
| `constraint_type` | string | `"implication"`, `"inverse_implication"`, `"negation"`, or `"all"` |
| `rule_source` | string | `"legacy"` (original), `"augmented"` (LLM-generated only), or `"hybrid"` (both) |
| `epochs` | int | Number of training epochs |
| `batch_size` | int | Batch size for fact supervision |
| `lr` | float | Learning rate |
| `accumulation_steps` | int | Gradient accumulation steps |
| `logic_weight` | float | Weight for the logic loss term |
| `factual_weight` | float | Weight for the factual NLL loss term |
| `quantization` | bool | Enable 4-bit NF4 quantization via BitsAndBytes |
| `lora_r` | int | LoRA rank |
| `wandb` | bool | Enable Weights & Biases logging |
| `parallel` | bool | Enable multi-GPU DDP training |
| `val_interval` | int | Validate every N epochs |

**LTN-specific fields:**

| Field | Type | Description |
|---|---|---|
| `rule_batch_size` | int | Number of grounded rules per logic batch |
| `sat_agg_p` | float | Exponent for the pMeanError aggregation |
| `implication_floor` | float | Floor value for Reichenbach implication |
| `prob_chunk_size` | int | Chunk size for batching probability queries (GPU memory) |
| `max_train_rules` | int | Max grounded rules per epoch (`-1` for all) |

**Hybrid-specific fields:**

| Field | Type | Description |
|---|---|---|
| `sdd_weight` | float | Weight for the SDD loss within the logic loss |
| `ltn_weight` | float | Weight for the LTN loss within the logic loss |

## Training

### SDD Training

Train with Sentential Decision Diagram-based semantic loss:

```bash
python run.py --config configs/train_llama32_1b_sdd.json
```

### LTN Training

Train with Logic Tensor Network fuzzy logic loss:

```bash
python run.py --config configs/train_llama32_1b_ltn.json
```

### Hybrid Training

Train with both SDD and LTN losses combined:

```bash
python run.py --config configs/train_llama32_1b_hybrid.json
```

### Command-Line Overrides

Any config field can be overridden via CLI arguments:

```bash
python run.py \
  --config configs/train_llama32_1b_ltn.json \
  --logic_backend ltn \
  --logic_weight 5.0 \
  --factual_weight 1.0 \
  --constraint_type all \
  --rule_source hybrid \
  --run_name my-ltn-experiment
```

### Weights & Biases (Optional)

To enable W&B logging, set `"wandb": true` in the config file and provide a `--run_name`:

```bash
wandb login
python run.py --config configs/train_llama32_1b_ltn.json --run_name ltn-run-1
```

### Checkpoints

Model checkpoints (LoRA adapters) are saved to `checkpoints/epoch-{N}/` after each epoch. These are standard Hugging Face PEFT adapters and can be loaded for evaluation or further training.

## Evaluation

### Base Model (No Fine-Tuning)

Evaluate the pretrained LLM without any logic-aware training:

```bash
python run.py --config configs/eval_llama32_1b_base.json
```

### Fine-Tuned Models

Evaluate a trained checkpoint by specifying the adapter path in the config or via CLI:

```bash
python run.py \
  --config configs/eval_llama32_1b_ltn.json \
  --checkpoint checkpoints/epoch-3/ltn_checkpoint
```

Similarly for SDD and hybrid:

```bash
python run.py \
  --config configs/eval_llama32_1b_sdd.json \
  --checkpoint checkpoints/epoch-3/sdd_checkpoint

python run.py \
  --config configs/eval_llama32_1b_hybrid.json \
  --checkpoint checkpoints/epoch-3/hybrid_checkpoint
```

### Evaluation Output

Results are saved as JSON files in the `results/` directory with the naming pattern:

```
results/{backend}_{model}_{timestamp}.json
```

Each result file contains per-prompt and aggregate scores for:
- **Accuracy** - factual correctness against silver labels
- **Consistency** - adherence to forward implication constraints
- **Inverse Consistency** - adherence to contrapositive constraints
- **Negation Consistency** - agreement between a fact and its negation
- **Satisfiability** - proportion of constraint pairs jointly satisfied

### Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Fraction of facts where the model's yes/no answer matches the silver label |
| Consistency | For every true antecedent A in A implies B, checks if B is also predicted true |
| Inverse Consistency | For every predicted-false consequent B in A implies B, checks if A is also predicted false |
| Negation Consistency | For pairs (fact, negated fact), checks that the model does not assign the same label to both |
| Satisfiability | Fraction of (antecedent, consequent) pairs where the joint assignment satisfies the constraint formula |

## Reproducing All Experiments

To reproduce the full set of results reported in our paper, run the following sequence. Each training run takes roughly 20-40 minutes on a single GPU (e.g. NVIDIA A100 or RTX 4090) with 4-bit quantization.

```bash
# Step 1: Train all three configurations
python run.py --config configs/train_llama32_1b_sdd.json
python run.py --config configs/train_llama32_1b_ltn.json
python run.py --config configs/train_llama32_1b_hybrid.json

# Step 2: Evaluate the base model (no checkpoint needed)
python run.py --config configs/eval_llama32_1b_base.json

# Step 3: Evaluate each trained model (update checkpoint paths as needed)
python run.py --config configs/eval_llama32_1b_sdd.json \
  --checkpoint checkpoints/epoch-3/sdd_checkpoint

python run.py --config configs/eval_llama32_1b_ltn.json \
  --checkpoint checkpoints/epoch-3/ltn_checkpoint

python run.py --config configs/eval_llama32_1b_hybrid.json \
  --checkpoint checkpoints/epoch-3/hybrid_checkpoint
```

Check the `results/` directory for the output JSON files after each evaluation run.

## Jupyter Notebook

An interactive walkthrough of the LTN training pipeline is available in:

```
LTN_LLM_training_for_first_order_logic.ipynb
```

This notebook demonstrates the fuzzy logic operators, rule grounding, probability extraction, and loss computation step by step.

## Troubleshooting

| Problem | Solution |
|---|---|
| `RuntimeError: CUDA is required` | Make sure you have an NVIDIA GPU and that PyTorch was installed with CUDA support. Run `python -c "import torch; print(torch.cuda.is_available())"` to verify. |
| `OutOfMemoryError` during training | Reduce `batch_size` or `rule_batch_size` in the config. Ensure `quantization` is set to `true`. Lower `prob_chunk_size` for LTN runs. |
| `NaN` losses during SDD training | This can happen with unsatisfiable constraint formulas. The trainer includes edge-case handling, but if it persists, try using `"constraint_type": "implication"` instead of `"all"`. |
| Poor LTN convergence | Increase `logic_weight` (e.g. 5.0 or higher). With low weights the logic loss signal is too weak relative to the factual loss. |
| Hugging Face authentication error | Run `huggingface-cli login` and make sure you have accepted the LLaMA license at [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B). |
| `ModuleNotFoundError: pysdd` | Install the PySDD package: `pip install pysdd`. On some systems you may need a C compiler (gcc/clang). |
| Slow training without quantization | Set `"quantization": true` in the config to enable 4-bit NF4 quantization. This significantly reduces memory and compute requirements. |

## Citation

If you use this codebase, please cite the original LoCo-LM paper:

```bibtex
@inproceedings{calanzone2025loco,
  title={Logically Consistent Language Models via Neuro-Symbolic Integration},
  author={Diego Calanzone and Stefano Teso and Antonio Vergari},
  booktitle={ICLR},
  year={2025},
  url={https://arxiv.org/abs/2409.13724},
}
```

## License

This project is released under the [MIT License](LICENSE).
