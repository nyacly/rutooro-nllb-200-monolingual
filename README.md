# Rutooro NLLB-200 Monolingual Corpus

This repository provides scripts and instructions for preparing and training a monolingual language model for Rutooro text using HuggingFace Transformers and Datasets.

## Repository Structure

- `data/raw/` – Place your raw `.txt` files here.
- `data/processed/` – Output directory for cleaned and deduplicated sentences.
- `data/datasets/` – Output HuggingFace `DatasetDict` saved with `save_to_disk()`.
- `scripts/` – Contains data processing and training scripts.
- `notebooks/` – Jupyter notebooks including the Colab training workflow.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your raw text files to `data/raw/` (or pass a different path).

3. Run the cleaning and splitting script:
   ```bash
   python scripts/clean_and_split.py --raw_dir data/raw --processed_dir data/processed
   ```

4. Create the dataset:
   ```bash
   python scripts/create_dataset.py --processed_file data/processed/cleaned_sentences.txt --dataset_dir data/datasets
   ```

5. Train the model:
   ```bash
   python scripts/train_nllb.py --dataset_dir data/datasets --output_dir nllb_rutooro_finetuned
   ```

Checkpoints and logs can be adjusted inside `scripts/train_nllb.py`.

## Using Google Colab and Google Drive

You can run the entire pipeline in Colab so that all data and model outputs are stored on your Google Drive.

1. Open the training notebook in Colab:
   [Open in Colab](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/rutooro-nllb-200-monolingual/blob/main/notebooks/rutooro_mlm_training.ipynb)

2. Mount your Drive inside Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Choose where to store your data and models on Drive by editing the paths in the notebook:

   ```python
   from pathlib import Path

   RAW_DATA_DIR = Path('/content/drive/MyDrive/rutooro-mlm/data/raw')
   PROCESSED_DATA_DIR = Path('/content/drive/MyDrive/rutooro-mlm/data/processed')
   DATASETS_DIR = Path('/content/drive/MyDrive/rutooro-mlm/data/datasets')
   MODEL_DIR = Path('/content/drive/MyDrive/rutooro-mlm/models/nllb_rutooro_finetuned')
   ```

4. Run the cells to clean the data, create the dataset, and fine‑tune the model. All outputs will be written to the directories above on your Drive so they persist after the Colab session ends.

You can also execute the scripts directly in a Colab cell by specifying paths:

```python
!python scripts/clean_and_split.py --raw_dir $RAW_DATA_DIR --processed_dir $PROCESSED_DATA_DIR
!python scripts/create_dataset.py --processed_file $PROCESSED_DATA_DIR/cleaned_sentences.txt --dataset_dir $DATASETS_DIR
!python scripts/train_nllb.py --dataset_dir $DATASETS_DIR --output_dir $MODEL_DIR
```

After training you can download the model or continue working with it from Drive.
