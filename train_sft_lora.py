import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA SFT baseline for Qwen2.5 on AI-MO/NuminaMath-CoT."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-name", default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--output-dir", default="./qwen2.5-1.5b-numina-sft-lora")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script expects a GPU.")

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    train_count = min(args.train_samples, len(dataset["train"]))

    train_dataset = dataset["train"].shuffle(seed=args.seed).select(range(train_count))
    eval_dataset = dataset["test"]

    train_dataset = train_dataset.map(
        to_messages, remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(to_messages, remove_columns=eval_dataset.column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "sdpa",
        },
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        packing=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()

