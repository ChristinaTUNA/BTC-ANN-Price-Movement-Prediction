# BTC ANN Price Movement Prediction

A lightweight project for predicting Bitcoin next-day price movement (bullish/bearish) using a feedforward Artificial Neural Network (ANN). The workflow covers data acquisition, cleaning, feature engineering, windowed dataset preparation, model training and evaluation.

## Overview
- Objective: Classify next 7-days BTC movement based on historical daily OHLCV and technical indicators data.
- Approach: LSTM trained on engineered features from sliding windows.
- Artifacts: Trained models (`.h5`), processed dataset CSVs, and Jupyter notebooks.

## Repository Structure
- `01_data_acq.ipynb`: Data acquisition, cleaning, and feature engineering. Produces `btc_processed.csv`.
- `02_model.ipynb`: Windowing, model definition, training, evaluation, and saving `.h5` models.
- `btc_2015_2024.csv`: Raw daily BTC OHLCV dataset (source-prepared).
- `btc_processed.csv`: Cleaned and engineered dataset ready for modeling.
- `btc_dense_model.h5` / `best_model_temp.h5`: Saved Keras ANN weights.

## Requirements
This project uses Python with common data science libraries.

Suggested environment (adjust versions if needed):
- Python 3.9+
- numpy, pandas, scikit-learn
- tensorflow or keras
- matplotlib, seaborn

Install packages (if using `pip`):
```zsh
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Usage
1. Open the notebooks in VS Code or Jupyter.
2. Run `01_data_acq.ipynb` to generate `btc_processed.csv` (skip if already present).
3. Run `02_model.ipynb` to train/evaluate the ANN and save a model to `.h5`.

### Quick Start (headless)
While the project is notebook-first, you can quickly inspect datasets:
```zsh
wc -l btc_2015_2024.csv btc_processed.csv
```

## Model Details
- Architecture: Dense layers (feedforward ANN) with regularization and standard activations.
- Input: Sliding window features derived from daily OHLCV and engineered indicators.
- Target: Binary label for next 7-days trend (bullish vs bearish).
- Metrics: Accuracy, precision/recall, confusion matrix (see `02_model.ipynb`).

## Results & Reproducibility
- Trained weights are stored in `btc_dense_model.h5` and `best_model_temp.h5`.
- Due to stochastic training, results may vary across runs; set random seeds in the notebook cells if you need strict reproducibility.

## Notes
- The raw dataset `btc_2015_2024.csv` spans multiple years; ensure the date parsing and any timezone handling match your locale.
- If GPU acceleration is available, TensorFlow will detect it automatically; otherwise training runs on CPU.