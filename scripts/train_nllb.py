#!/usr/bin/env python
from pathlib import Path
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train(dataset_dir: Path, model_name: str, output_dir: Path) -> None:
    """Fine-tune the NLLB model on the provided dataset."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} not found. Run create_dataset.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model trained and saved to {output_dir}")


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB on Rutooro data")
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/datasets"), help="Directory containing HF dataset")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M", help="Base model name")
    parser.add_argument("--output_dir", type=Path, default=Path("nllb_rutooro_finetuned"), help="Where to save the fine-tuned model")
    args = parser.parse_args()

    train(args.dataset_dir, args.model_name, args.output_dir)


if __name__ == '__main__':
    main()
