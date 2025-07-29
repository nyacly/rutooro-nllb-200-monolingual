#!/usr/bin/env python
import random
from pathlib import Path
from datasets import Dataset, DatasetDict

PROCESSED_FILE = Path('data/processed/cleaned_sentences.txt')
DATASET_DIR = Path('data/datasets')


def main():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"{PROCESSED_FILE} not found. Run clean_and_split.py first.")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    random.shuffle(sentences)
    split_idx = int(0.9 * len(sentences))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    train_ds = Dataset.from_dict({'text': train_sentences})
    val_ds = Dataset.from_dict({'text': val_sentences})
    ds = DatasetDict({'train': train_ds, 'validation': val_ds})
    ds.save_to_disk(DATASET_DIR)

    print(f"Dataset saved to {DATASET_DIR}")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Validation sentences: {len(val_sentences)}")


if __name__ == '__main__':
    main()
