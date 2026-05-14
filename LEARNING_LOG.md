# Learning Log

This log tracks my learning process while building a LoRA supervised fine-tuning
workflow for math reasoning models.

## 2026-05-13

### What I Worked On

- Set up a LoRA SFT baseline for `Qwen/Qwen2.5-1.5B-Instruct`.
- Used the `AI-MO/NuminaMath-CoT` dataset from Hugging Face.
- Converted math samples into chat-style user/assistant messages.
- Ran a small smoke test to check that the training pipeline works end to end.

### What I Learned

- LoRA fine-tuning updates small adapter weights instead of all base model
  parameters.
- TRL `SFTTrainer` can train on chat-style message data.
- PEFT adapters can be saved separately from the original base model.
- Small smoke tests are useful before running longer training jobs.

### Notes

- The current script is LoRA SFT, not full QLoRA yet.
- The next useful step is to add an inference script that loads the base model
  together with the saved LoRA adapter.

