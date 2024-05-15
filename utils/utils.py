import logging
from typing import Literal, Any

import numpy as np
import numbers

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from scoring.scores import SCORES

logger = logging.getLogger(__name__)


def init_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        force=True,
    )


def get_null_count(data: pd.DataFrame) -> None:
    logger.info(f"Null count: {np.sum(data.isna().sum())}")


def remove_duplicates(data: pd.DataFrame, verbose: bool = True) -> None:
    """Drop duplicates inplace."""
    duplicates = data[data.duplicated()]
    if verbose:
        logger.info(f"Dropped {len(duplicates)} duplicates.")
    data.drop_duplicates(inplace=True)


def compute_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    regression: bool,
    y_pp: np.ndarray | None = None,
    average: str = "binary",
    verbose: bool = True,
) -> dict[str, numbers.Real]:
    metrics = {}

    if regression:
        metrics["mae"] = mean_absolute_error(y_true, y_hat)
        metrics["mse"] = mean_squared_error(y_true, y_hat)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["r2"] = r2_score(y_true, y_hat)
    else:
        metrics["accuracy"] = accuracy_score(y_true, y_hat)

        if len(np.unique(y_true)) == 2:
            metrics["precision"] = precision_score(y_true, y_hat, average="binary")
            metrics["recall"] = recall_score(y_true, y_hat, average="binary")
            metrics["f1"] = f1_score(y_true, y_hat, average="binary")

            if y_pp is not None:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pp)
                metrics["pr_auc"] = average_precision_score(y_true, y_pp)
        else:
            metrics["precision"] = precision_score(y_true, y_hat, average=average)
            metrics["recall"] = recall_score(y_true, y_hat, average=average)
            metrics["f1"] = f1_score(y_true, y_hat, average=average)

            if y_pp is not None and average == "macro":
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_pp, average=average, multi_class="ovr"
                )
    if verbose:
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

    return metrics


def flood_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> numbers.Real:
    """
    Custom scoring function to use for
    scikit-learn methods for optimising hyperparameters.
    """
    score = sum(
        SCORES[pred - 1, true - 1]
        for pred, true in zip(np.round(y_pred).astype(int) + 1, y_true.astype(int) + 1)
    )
    return score


def tune_decision_boundary(
    model: Any,
    tune_metric: Literal["precision", "recall"],
    X: np.ndarray,
    y: np.ndarray,
    upper_qty: numbers.Real = 0.9,
    return_preds: bool = True,
) -> tuple[numbers.Real, numbers.Real] | np.ndarray:
    """Method to perform post-processing by using precision and recall to tune decision boundary."""

    # this will generate an error if our model can't predict probabilities, handle later
    y_proba = model.predict_proba(X)
    precision, recall, thresholds = precision_recall_curve(
        y_true=y, probas_pred=y_proba[:, 1]
    )
    df_pr = pd.DataFrame(
        {"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds}
    )

    # now begin tuning
    aux_metric = "precision" if tune_metric == "recall" else "recall"
    thold_idx = (
        df_pr[df_pr[tune_metric] >= upper_qty]
        .sort_values(aux_metric, ascending=False)
        .index[0]
    )
    thold = df_pr.loc[thold_idx, "threshold"]

    if return_preds:
        return (y_proba[:, 1] >= thold).astype(int)

    return thold, thold_idx


def binning(y: pd.Series | np.ndarray, q: int = 0) -> np.ndarray | pd.Series:
    """
    Function for creating q bins for y (categorising a continuous variable)
    """
    if q == 0:
        q = int(np.sqrt(len(y)))
    y_bins = pd.Series(y)
    # using pandas qcut for quantile division of the data
    y_bins = pd.qcut(y_bins, q=q, labels=np.arange(q), duplicates="drop")
    return y_bins
