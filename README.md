# Rutooro NLLB-200 Monolingual Corpus

This repository provides scripts and instructions for preparing and training a monolingual language model for Rutooro text using HuggingFace Transformers and Datasets.

## Repository Structure

- `data/raw/` – Place your raw `.txt` files here.
- `data/processed/` – Output directory for cleaned and deduplicated sentences.
- `data/datasets/` – Output HuggingFace `DatasetDict` saved with `save_to_disk()`.
- `scripts/` – Contains data processing and training scripts.
- `notebooks/` – Optional Jupyter notebooks for exploratory data analysis and manual cleaning.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your raw text files to `data/raw/`.

3. Run the cleaning and splitting script:
   ```bash
   python scripts/clean_and_split.py
   ```

4. Create the dataset:
   ```bash
   python scripts/create_dataset.py
   ```

5. Train the model:
   ```bash
   python scripts/train_nllb.py
   ```

Checkpoints and logs can be adjusted inside `scripts/train_nllb.py`.
