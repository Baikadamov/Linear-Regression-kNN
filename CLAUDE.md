# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University (Narxoz) machine learning coursework consisting of Jupyter notebooks that cover supervised learning topics. Notebooks are written in Python with commentary in Russian.

## Structure

- **Root notebooks**: Linear regression (`Linear-regression.ipynb` on WineQT.csv) and kNN classification (`Classification.ipynb` on sklearn's wine dataset)
- **`linear/`**: Extended linear regression analysis (`Linear2.ipynb` on housing.csv)
- **`polinomial/`**: Polynomial regression with regularization and cross-validation (`task2.ipynb`)
- **Task 3*/**: Logistic regression, binary classification, regularization, cross-validation (in progress)
- **Task 4*/**: Decision trees and random forests for classification and regression (in progress)

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Python 3.12 with a `.venv` virtual environment. Key libraries: scikit-learn, pandas, numpy, matplotlib, seaborn.

## Running Notebooks

```bash
jupyter nbconvert --to notebook --execute <notebook>.ipynb
```

Or open in Jupyter/VS Code and run cells interactively.

## Datasets

- `WineQT.csv` — red wine quality (1143 rows, target: `quality`), used by root linear regression notebook
- `housing.csv` — housing data, used by `linear/Linear2.ipynb`
- `sklearn.datasets.load_wine()` — built-in wine classification (178 rows, 3 classes), used by `Classification.ipynb`

## Conventions

- Each task folder contains a `task*.txt` with the assignment requirements and a report markdown file (`report_*.md`)
- Notebooks follow a consistent pipeline: EDA (info/describe/correlations/boxplots) → preprocessing (outliers, scaling with StandardScaler) → model training with train_test_split (test_size=0.2, random_state=42) → evaluation metrics
- Regression metrics: MAE, MSE, RMSE, R²
- Classification metrics: accuracy, confusion matrix, classification report, AUC-ROC
