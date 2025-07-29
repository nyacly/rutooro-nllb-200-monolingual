#!/usr/bin/env python
import random
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict


def create_dataset(processed_file: Path, dataset_dir: Path) -> None:
    """Create a HuggingFace dataset from a cleaned sentence file."""
    if not processed_file.exists():
        raise FileNotFoundError(f"{processed_file} not found. Run clean_and_split.py first.")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    random.shuffle(sentences)
    split_idx = int(0.9 * len(sentences))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    train_ds = Dataset.from_dict({'text': train_sentences})
    val_ds = Dataset.from_dict({'text': val_sentences})
    ds = DatasetDict({'train': train_ds, 'validation': val_ds})
    ds.save_to_disk(dataset_dir)

    print(f"Dataset saved to {dataset_dir}")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Validation sentences: {len(val_sentences)}")


def main():
    parser = argparse.ArgumentParser(description="Create HF dataset from cleaned text")
    parser.add_argument("--processed_file", type=Path, default=Path("data/processed/cleaned_sentences.txt"), help="Path to cleaned sentence file")
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/datasets"), help="Output directory for dataset")
    args = parser.parse_args()

    create_dataset(args.processed_file, args.dataset_dir)


if __name__ == '__main__':
    main()
