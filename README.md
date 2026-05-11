# LLM LoRA SFT

Supervised fine-tuning baseline for `Qwen/Qwen2.5-1.5B-Instruct` on the
`AI-MO/NuminaMath-CoT` dataset using LoRA adapters.

This repository trains a lightweight PEFT adapter instead of updating all model
weights. The output can be loaded together with the original base model for
inference or further experiments.

## What This Project Does

- Loads `AI-MO/NuminaMath-CoT` from Hugging Face Datasets.
- Converts each math sample into chat-style messages:
  - user: problem
  - assistant: solution
- Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` with TRL `SFTTrainer`.
- Uses PEFT LoRA on attention and MLP projection layers.
- Saves only the LoRA adapter files.

## Important Note

This is a LoRA SFT project, not a full QLoRA implementation yet.

The current script does not use 4-bit quantized loading with bitsandbytes. To
turn this into QLoRA, add a `BitsAndBytesConfig` with 4-bit loading and prepare
the model for k-bit training.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- train_sft_lora.py
|-- inspect_dataset.py
`-- .gitignore
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your GPU if needed. See the official
PyTorch installation selector for the correct command.

## Check the Dataset

```bash
python inspect_dataset.py
```

## Train

Default training command:

```bash
python train_sft_lora.py
```

Small smoke test:

```bash
python train_sft_lora.py \
  --train-samples 200 \
  --max-steps 10 \
  --save-steps 10 \
  --eval-steps 10 \
  --logging-steps 1 \
  --output-dir ./qwen2.5-1.5b-numina-smoke
```

Useful arguments:

```bash
python train_sft_lora.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name AI-MO/NuminaMath-CoT \
  --output-dir ./qwen2.5-1.5b-numina-sft-lora \
  --train-samples 5000 \
  --max-length 1024 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4
```

## LoRA Configuration

The training script uses:

- rank `r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `bias="none"`
- task type `CAUSAL_LM`

Target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

## Output

The output directory contains adapter files such as:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

The adapter should be used together with the base model:

```text
Qwen/Qwen2.5-1.5B-Instruct + saved LoRA adapter
```

## Push to GitHub

From inside this folder:

```bash
git init
git add .
git commit -m "Initial LoRA SFT baseline"
git branch -M main
git remote add origin https://github.com/binbingwu/LLM-LoRA-SFT.git
git push -u origin main
```
