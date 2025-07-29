#!/usr/bin/env python
import os
import re
import argparse
from pathlib import Path
from nltk import sent_tokenize


def clean_and_split(raw_dir: Path, processed_dir: Path, output_file: Path) -> None:
    """Clean raw text files and split them into unique sentences."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sentences = []
    files_processed = 0
    for txt_path in raw_dir.glob('*.txt'):
        files_processed += 1
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
        sents = sent_tokenize(text)
        sentences.extend(sents)
        print(f"Processed {txt_path} -> {len(sents)} sentences")

    unique_sentences = list(dict.fromkeys(sentences))
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for s in unique_sentences:
            out_f.write(s + '\n')
    print(f"Processed {files_processed} files")
    print(f"Total sentences: {len(sentences)}")
    print(f"Unique sentences: {len(unique_sentences)}")
    print(f"Saved cleaned sentences to {output_file}")


def clean_text(text: str) -> str:
    """Simple cleaning for Rutooro text."""
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Clean Rutooro text and split into sentences")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"), help="Directory containing raw .txt files")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"), help="Directory to store processed outputs")
    parser.add_argument("--output_file", type=Path, default=None, help="Path to save cleaned sentences")
    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = args.processed_dir / "cleaned_sentences.txt"

    clean_and_split(args.raw_dir, args.processed_dir, output_file)


if __name__ == '__main__':
    main()
