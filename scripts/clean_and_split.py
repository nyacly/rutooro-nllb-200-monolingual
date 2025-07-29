#!/usr/bin/env python
import os
import re
from pathlib import Path
from nltk import sent_tokenize

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
OUTPUT_FILE = PROCESSED_DIR / 'cleaned_sentences.txt'


def clean_text(text: str) -> str:
    """Simple cleaning for Rutooro text."""
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.strip()
    return text


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sentences = []
    files_processed = 0
    for txt_path in RAW_DIR.glob('*.txt'):
        files_processed += 1
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
        sents = sent_tokenize(text)
        sentences.extend(sents)
        print(f"Processed {txt_path} -> {len(sents)} sentences")

    unique_sentences = list(dict.fromkeys(sentences))
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for s in unique_sentences:
            out_f.write(s + '\n')
    print(f"Processed {files_processed} files")
    print(f"Total sentences: {len(sentences)}")
    print(f"Unique sentences: {len(unique_sentences)}")
    print(f"Saved cleaned sentences to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
