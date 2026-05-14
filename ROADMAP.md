# Roadmap

This roadmap tracks the next learning and engineering steps for the LoRA SFT
project.

## Completed

- [x] Create a baseline LoRA SFT training script.
- [x] Load `AI-MO/NuminaMath-CoT` with Hugging Face Datasets.
- [x] Format math problems and solutions as chat-style messages.
- [x] Run a small smoke test to verify the training pipeline.
- [x] Document the current LoRA configuration in the README.

## Next

- [ ] Add an inference script for loading the saved LoRA adapter.
- [ ] Add sample prompts and generated outputs.
- [ ] Record training loss observations from short experiments.
- [ ] Try different LoRA ranks such as `r=8`, `r=16`, and `r=32`.
- [ ] Add a small evaluation workflow for math answer quality.

## Future Experiments

- [ ] Convert the project from LoRA SFT to QLoRA with 4-bit loading.
- [ ] Compare training speed and memory usage across settings.
- [ ] Try a larger training subset from NuminaMath.
- [ ] Test another instruction model as the base model.
- [ ] Write a short summary of what worked, what failed, and what to improve.

