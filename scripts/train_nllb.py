#!/usr/bin/env python
from pathlib import Path
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

DATASET_DIR = Path('data/datasets')
MODEL_NAME = 'facebook/nllb-200-distilled-600M'
OUTPUT_DIR = Path('nllb_rutooro_finetuned')


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"{DATASET_DIR} not found. Run create_dataset.py first.")

    dataset = load_from_disk(DATASET_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy='epoch',
        save_strategy='epoch',
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
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model trained and saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
