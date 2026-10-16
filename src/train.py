"""
Training functions for Case 1.

train_logistic_regression() and train_xgboost() are yours to implement.
apply_baseline_rule() is provided, and you will use it to benchmark your model
against the bank's current manual process.

Both training functions take the same three arguments and return the same kind
of object, so the rest of the notebook treats them interchangeably.
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.api as sms
import xgboost as xgb                        # Data visualization library, extension of matplotlib  
import random                                           # Randomly generating numbers
import math   

from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import qqplot

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import scipy.stats as stats                             # Fit of distribution plot
from sklearn.model_selection import train_test_split    # Split train-/testset     
from sklearn.tree import DecisionTreeClassifier         # Modeling CART Decision Tree  
from sklearn import metrics                             # Performance metrics
from sklearn.metrics import accuracy_score              # Accuracy of a model
from sklearn.metrics import classification_report       # Performance report of a classification model
#import graphviz as gv                                   # Needed to graph Decision Tree
#import pydotplus                                        # Needed to graph Decision Tree
from IPython.display import Image                       # Needed to graph Decision Tree
#from sklearn.externals.six import StringIO              # Needed to graph Decision Tree
from sklearn.tree import export_graphviz  

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
):
    """
    Train a Logistic Regression classifier on X_train/y_train.

    Receives/Args:
    =============
        X_train : DataFrame of model-ready features. Already encoded and
                  imputed, so there are no NaNs and no text columns.
        y_train : integer Series of labels, 1 = fraud, 0 = legitimate.
        config  : the loaded config.yaml, as a dictionary. Holds random_state,
                  the feature lists, the premium definition and the baseline
                  rule, e.g. via: seed = config["random_state"]

                  Note: config deliberately contains NO model hyperparameters.
                  Picking those is part of the exercise. If you would rather
                  read them from config than hardcode them here, add your own
                  section to config.yaml and read it from this function.

    Returns:
    =========
        a fitted model with a .predict_proba() method.
        use the function e.g. the following way:
            lr = train_logistic_regression(X_train_imp, y_train, CONFIG)

    Things to consider:
    - The fraud rate is ~0.15%. How should your model handle that?
    - Logistic Regression is sensitive to feature scale. Does that matter here?
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config["random_state"],
        ),
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
):
    """
    Train an XGBoost classifier on X_train/y_train.

    Receives/Args:
    =============
        X_train : DataFrame of model-ready features. Already encoded and
                  imputed, so there are no NaNs and no text columns.
        y_train : integer Series of labels, 1 = fraud, 0 = legitimate.
        config  : the loaded config.yaml, as a dictionary. Holds random_state,
                  the feature lists, the premium definition and the baseline
                  rule, e.g. via: seed = config["random_state"]

                  Note: config deliberately contains NO model hyperparameters.
                  Picking those is part of the exercise. If you would rather
                  read them from config than hardcode them here, add your own
                  section to config.yaml and read it from this function.

    Returns:
    =========
        a fitted model with a .predict_proba() method.
        use the function e.g. the following way:
            xgb = train_xgboost(X_train_imp, y_train, CONFIG)

    Things to consider:
    - The fraud rate is ~0.15%. XGBoost has a parameter specifically for this.
    - What evaluation metric makes sense to optimise during training?
    """
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0:
        raise ValueError("y_train must contain at least one fraud example.")

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=negative_count / positive_count,
        random_state=config["random_state"],
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model



def apply_baseline_rule(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Apply the bank's current manual rule and return a boolean Series of flags.

    This one is provided. It is the benchmark every model result should be
    compared against: a model that cannot beat it has not earned its place.

    The rule, adapted for US data:
        flag if amount > threshold AND the merchant is in a foreign state.

    Receives/Args:
    =============
        df     : the cleaned DataFrame, NOT the feature matrix. The rule needs
                 the original 'amount' and 'is_foreign_state' columns, which do
                 not survive into X in their raw form.
        config : the loaded config.yaml. Reads config["baseline_rule"], which
                 holds amount_threshold and requires_foreign_state.

    Returns:
    =========
        a boolean Series, True where the rule would flag the transaction.
        use the function e.g. the following way:
            rule_flags = apply_baseline_rule(df_test, CONFIG)

    Things to consider:
    - Check what is_foreign_state actually contains before trusting the result
      (notebook 01 covers this). If that column is not what you expect, this
      rule is not measuring what it claims to, and neither is your benchmark.
    """
    rule = config["baseline_rule"]
    flagged = (df["amount"] > rule["amount_threshold"])
    if rule["requires_foreign_state"] and "is_foreign_state" in df.columns:
        flagged = flagged & df["is_foreign_state"]
    return flagged
