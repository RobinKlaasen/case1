"""Load Case 1 data, train both classifiers, and predict on a held-out set."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_features import build_features, load_config
from src.clean import clean
from src.load import load_case1_data
from src.train import apply_baseline_rule, train_logistic_regression, train_xgboost


CONFIG = load_config(PROJECT_ROOT / "config.yaml")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROBABILITY_THRESHOLD = 0.30


def plot_roc_curves(
    y_test: pd.Series,
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Plot ROC curves and return each model's ROC AUC."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color="black",
            label="Random ranking")

    auc_scores = {}
    for model_name in ("logistic", "xgboost"):
        probabilities = predictions[f"{model_name}_fraud_probability"]
        fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
        auc_score = metrics.auc(fpr, tpr)
        auc_scores[model_name] = float(auc_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.3f})")

    ax.set_title("Fraud ranking performance on the held-out test set")
    ax.set_ylabel("Fraud detected (true positive rate)")
    ax.set_xlabel("Legitimate transactions incorrectly flagged (false positive rate)")
    ax.legend(loc="lower right")
    fig.text(
        0.5,
        0.01,
        "AUC summarizes ranking across all probability cutoffs; the dashed line is random ranking.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return auc_scores


def plot_xgboost_importances(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Print feature importance rankings and plot the top 20 features."""
    importances = model.feature_importances_
    importance_table = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("XGBoost feature ranking:")
    print(importance_table.to_string(index=False))

    top_features = importance_table.head(20).sort_values("importance")
    ax = top_features.plot.barh(
        x="feature",
        y="importance",
        legend=False,
        figsize=(9, 7),
        color="firebrick",
    )
    ax.set_title("XGBoost: top 20 features by model importance")
    ax.set_xlabel("Relative importance (higher means more used by the model)")
    ax.set_ylabel("Feature")
    ax.text(
        0,
        -0.12,
        "Importance describes model reliance, not a causal effect.",
        transform=ax.transAxes,
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return importance_table


def plot_logistic_coefficients(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Show standardized logistic-regression coefficients and plot the top 20."""
    classifier = model.named_steps["logisticregression"]
    coefficient_table = (
        pd.DataFrame(
            {"feature": feature_names, "coefficient": classifier.coef_[0]}
        )
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    print(
        "Logistic coefficients (positive pushes toward fraud; "
        "negative pushes toward legitimate):"
    )
    print(coefficient_table[["feature", "coefficient"]].to_string(index=False))

    top_features = coefficient_table.head(20).sort_values("coefficient")
    colors = np.where(top_features["coefficient"] >= 0, "firebrick", "steelblue")
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Logistic regression: strongest standardized feature effects")
    plt.xlabel("Change in fraud log-odds per standardized feature unit")
    plt.ylabel("Feature")
    plt.figtext(
        0.5,
        0.01,
        "Positive coefficients raise the model's fraud score; negative coefficients lower it.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return coefficient_table


def plot_xgboost_shap(
    model: Any,
    X_test: pd.DataFrame,
    max_rows: int = 1_000,
    random_state: int = 42,
) -> tuple[Any, shap.Explanation]:
    """Plot a SHAP beeswarm using a reproducible sample of the test set."""
    X_explain = X_test.sample(
        n=min(max_rows, len(X_test)),
        random_state=random_state,
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_explain)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("XGBoost: how the top features influence fraud predictions")
    plt.gcf().text(
        0.5,
        0.01,
        "Each point is a test transaction; color shows feature value and horizontal position shows SHAP impact.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return explainer, shap_values


def train_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    test_size: float = 0.2,
    baseline_flags: pd.Series | None = None,
) -> dict[str, Any]:
    """Train both models and compare them at the same fixed cutoff on test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config["random_state"],
        stratify=y,
    )

    logistic_model = train_logistic_regression(X_train, y_train, config)
    xgboost_model = train_xgboost(X_train, y_train, config)

    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]
    xgboost_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

    predictions = pd.DataFrame(
        {
            "actual_is_fraud": y_test,
            "logistic_fraud_probability": logistic_probabilities,
            "logistic_prediction": (
                logistic_probabilities >= PROBABILITY_THRESHOLD
            ).astype("int8"),
            "xgboost_fraud_probability": xgboost_probabilities,
            "xgboost_prediction": (
                xgboost_probabilities >= PROBABILITY_THRESHOLD
            ).astype("int8"),
        },
        index=X_test.index,
    )
    if baseline_flags is not None:
        predictions["baseline_prediction"] = (
            baseline_flags.loc[X_test.index].astype("int8")
        )

    confusion_matrices = {
        name: pd.DataFrame(
            confusion_matrix(y_test, predictions[column], labels=[0, 1]),
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"],
        )
        for name, column in (
            ("logistic", "logistic_prediction"),
            ("xgboost", "xgboost_prediction"),
        )
    }
    if baseline_flags is not None:
        confusion_matrices["baseline"] = pd.DataFrame(
            confusion_matrix(
                y_test, predictions["baseline_prediction"], labels=[0, 1]
            ),
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"],
        )

    roc_auc = {
        name: float(
            metrics.roc_auc_score(
                y_test,
                predictions[f"{name}_fraud_probability"],
            )
        )
        for name in ("logistic", "xgboost")
    }
    amounts = X_test["amount"]
    amount_summaries = {}
    prediction_columns = {
        "logistic": "logistic_prediction",
        "xgboost": "xgboost_prediction",
    }
    if baseline_flags is not None:
        prediction_columns["baseline"] = "baseline_prediction"

    for name, prediction_column in prediction_columns.items():
        predicted_fraud = predictions[prediction_column].eq(1)
        actual_fraud = y_test.eq(1)
        amount_summaries[name] = {
            "Fraud caught (TP)": amounts[actual_fraud & predicted_fraud].sum(),
            "Fraud missed (FN)": amounts[actual_fraud & ~predicted_fraud].sum(),
            "Legitimate flagged (FP)": amounts[~actual_fraud & predicted_fraud].sum(),
            "Legitimate not flagged (TN)": amounts[~actual_fraud & ~predicted_fraud].sum(),
        }

    train_roc_auc = {
        name: float(
            metrics.roc_auc_score(
                y_train,
                model.predict_proba(X_train)[:, 1],
            )
        )
        for name, model in (
            ("logistic", logistic_model),
            ("xgboost", xgboost_model),
        )
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logistic_model": logistic_model,
        "xgboost_model": xgboost_model,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "predictions": predictions,
        "confusion_matrices": confusion_matrices,
        "amount_summaries": amount_summaries,
        "roc_auc": roc_auc,
        "train_roc_auc": train_roc_auc,
    }


def run_pipeline(
    sample_n: int | None = 200_000,
    raw_dir: str | Path = RAW_DIR,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Load full labelled data, train, predict, and show model diagnostics."""
    df_raw = load_case1_data(
        raw_dir,
        sample_n=sample_n,
        random_state=CONFIG["random_state"],
    )
    df = clean(df_raw)
    X, y = build_features(df, CONFIG)
    baseline_flags = apply_baseline_rule(df, CONFIG)
    del df_raw, df
    results = train_and_predict(X, y, CONFIG, baseline_flags=baseline_flags)
    if make_plots:
        results["roc_auc"] = plot_roc_curves(
            results["y_test"], results["predictions"]
        )
        feature_names = results["X_train"].columns.tolist()
        results["xgboost_feature_importances"] = plot_xgboost_importances(
            results["xgboost_model"], feature_names
        )
        results["logistic_coefficients"] = plot_logistic_coefficients(
            results["logistic_model"], feature_names
        )
        results["shap_explainer"], results["shap_values"] = plot_xgboost_shap(
            results["xgboost_model"],
            results["X_test"],
            random_state=CONFIG["random_state"],
        )
    return results


def print_model_comparison(results: dict[str, Any]) -> None:
    """Print the probability cutoffs and comparable test-set results."""
    print(f"Rows used: {len(results['X_train']) + len(results['X_test']):,}")
    print(f"Train rows: {len(results['X_train']):,}")
    print(f"Test rows: {len(results['X_test']):,}")
    print(
        f"Probability cutoff for both models: "
        f">= {results['probability_threshold']:.2f}"
    )
    print(
        "Confusion matrices below cover only the held-out test rows "
        "(20% of the rows used), not the full training sample."
    )

    comparison_rows = []
    for name in ("logistic", "xgboost", "baseline"):
        if name not in results["confusion_matrices"]:
            continue
        matrix = results["confusion_matrices"][name].to_numpy()
        tn, fp, fn, tp = matrix.ravel()
        comparison_rows.append(
            {
                "model": name,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Train AUC": results["train_roc_auc"].get(name),
                "Test AUC": results["roc_auc"].get(name),
            }
        )

    print("\nTest-set comparison (TN/FP/FN/TP; train AUC is diagnostic only):")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print("\nConfusion matrices (rows=actual; columns=predicted):")
    for name in ("logistic", "xgboost", "baseline"):
        if name in results["confusion_matrices"]:
            print(f"\n{name}:")
            print(results["confusion_matrices"][name])
    print("\nTransaction amounts by outcome (USD, summed transaction amounts):")
    amount_table = pd.DataFrame(results["amount_summaries"]).T
    formatted_amounts = amount_table.apply(
        lambda column: column.map(lambda amount: f"${amount:,.2f}")
    )
    print(formatted_amounts.to_string())


if __name__ == "__main__":
    results = run_pipeline()
    print_model_comparison(results)
    print(results["predictions"].head())
