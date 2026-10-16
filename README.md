# Case: Fraud Detection, Business Oriented 

Welcome. This folder contains everything you need to work through the BDAIM Case / Case 1.

---

## What you are working with

A real Kaggle dataset: [Financial Transactions Dataset: Analytics](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) by computingvictor.

Five files, ~1.4 GB total:

| File | Contents |
|---|---|
| `data/raw/transactions_data.csv` | ~13M transactions, 2010–2019 |
| `data/raw/users_data.csv` | 2,000 cardholders |
| `data/raw/cards_data.csv` | 6,146 cards |
| `data/raw/train_fraud_labels.json` | Labels for ~8.9M transactions |
| `data/raw/mcc_codes.json` | Merchant category code lookup |

Fraud rate: **~0.15%**, roughly 1 in 670 transactions.

---

## What is provided for you

- **`notebooks/01_data_exploration.ipynb`**: read this first. It walks through the raw data, explains every data quality issue, and shows the key findings that shape the modelling choices. This is context, not an exercise.
- **`src/load.py`, `src/clean.py`, `src/build_features.py`**: the data infrastructure modules. Fully implemented, you do not need to change them.
- **`src/train.py`**: contains `apply_baseline_rule()` (provided) and two training function stubs for you to implement: `train_logistic_regression()` and `train_xgboost()`.
- **`shared/load_raw.py`**: handles merging the five raw files (you do not need to worry about this).
- **`config.yaml`**: all parameters in one place. Feature lists, the premium cardholder definition, and the baseline rule. Changing which features to include for training etc means editing this file.

---

## Your working notebook

Open `notebooks/case1_starter.ipynb`. It loads the data and builds the feature matrix for you. Your job starts from there.

---

## The six sub-exercises in this case

1. **Choose the right metric.** Why is accuracy useless here? What should you use instead, and why?
2. **Evaluate and benchmark.** Measure the current manual rule (given in train.py). Train and evaluate the two new ML models (empty in train.py). Compare them.
3. **Think in business terms.** How much fraud value do the models catch? What does the false-positive rate look like for premium (Credit) cardholders? How should the committee interpret these numbers?
4. **Slice the results.** Does the model perform evenly across merchant categories and card types, or is it quietly failing somewhere?
5. **Explain the drivers.** Explain to your stakeholders what the most important drivers of the model are. You could, for example, use a SHAP plot.
6. **Plan the next steps.** What would be needed to make this model operational, and what is the effort? What can we already start doing today, based on these insights? If the same approach ships to three more markets next year, each with slightly different transaction data, what has to change?

---

## Time budget notes

Training on the full dataset takes roughly 4–5 minutes per model. If you are short on time, pass `sample_n=200_000` to `load_case1_data()` - the starter notebook shows how.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and its dependencies. Install uv first if you don't have it (see the uv docs), then, from this folder:

```bash
uv sync
```

This creates a `.venv` with everything listed in `pyproject.toml`. Then open VS Code, open `notebooks/case1_starter.ipynb`, and select the `.venv` kernel (`Select Kernel` > `Python Environments` > `.venv`).

To use JupyterLab instead:

```bash
uv run jupyter lab
```

