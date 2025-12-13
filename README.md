# Deepfake Detector

This project is a deepfake detection pipeline using TinyCNN.
It includes scripts to extract frames, crop faces, and train a model on Celeb-DF v2 images.

## Requirements

- Python 3.10
- PyTorch
- Pillow
- tqdm

## Usage

1. Extract frames: `python scripts/run_extract.py`
2. Train model: `python scripts/train_model.py`
3. Test dataset: `python scripts/test_dataset.py`

Dataset should be organized as:
