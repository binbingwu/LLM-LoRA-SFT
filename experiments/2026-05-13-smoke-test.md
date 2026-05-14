# Smoke Test: LoRA SFT on NuminaMath

## Goal

Verify that the LoRA supervised fine-tuning pipeline can run end to end before
starting longer training experiments.

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `AI-MO/NuminaMath-CoT`
- Training style: supervised fine-tuning with chat-formatted examples
- Adapter method: LoRA through PEFT
- Trainer: TRL `SFTTrainer`

## Command

```bash
python train_sft_lora.py \
  --train-samples 200 \
  --max-steps 10 \
  --save-steps 10 \
  --eval-steps 10 \
  --logging-steps 1 \
  --output-dir ./qwen2.5-1.5b-numina-smoke
```

## Result

The smoke test completed and produced a saved LoRA adapter directory.

Expected adapter files include:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`

## What I Learned

- A small training run is a good way to test the full pipeline quickly.
- The dataset needs to be converted into user/assistant messages before SFT.
- The saved adapter should be loaded together with the original base model.

## Next Step

Add an inference script so the adapter can be tested on new math prompts.

