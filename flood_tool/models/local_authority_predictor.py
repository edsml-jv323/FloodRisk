import logging
import numbers
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

from flood_tool.models.constant import TEST_SIZE, RDS
from flood_tool.models.predictor import Predictor
from utils.utils import compute_metrics

logger = logging.getLogger(__name__)


class LocalAuthorityPredictor(Predictor):
    """Class to handle classification of local authority based on coordinates."""

    def __init__(
        self,
        training_data: pd.DataFrame,
        scoring: Any,
        model_alias: str = "knn_classifier",
        tune_hyperparameters: bool = False,
        n_iter: int = 10,
    ):
        self.features = ["easting", "northing"]
        super().__init__(training_data, scoring, model_alias, tune_hyperparameters, n_iter)

    def prepare_data(self) -> None:
        """
        Label encoder is fit on the full dataset in order to stratify our train/test split.
        There's no data leakage as we are just mapping strings to integers.
        """
        # we only need labelled postcodes currently
        data = self.training_data

        self.X = data[self.features]
        self.y = data["localAuthority"]
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)

        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.values
        if isinstance(self.y, pd.Series):
            self.y = self.y.values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RDS, stratify=self.y
        )

    def build_pipeline(self) -> None:
        """Initialise pipeline with default hyperparameters"""
        self.model = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        algorithm="kd_tree", n_neighbors=4, weights="distance"
                    ),
                ),
            ]
        )

    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, scoring: str, n_iter: int
    ) -> None:
        logger.info("Tuning hyperparameters...")

        param_dist = {
            "scaler": [MinMaxScaler(), StandardScaler(), None],
            "model__n_neighbors": stats.randint(2, 10),
            "model__p": stats.randint(1, 10),
            "model__weights": ["uniform", "distance"],
            "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }

        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RDS)
        rand_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            cv=skfold,
            n_iter=n_iter,
            n_jobs=-1,
            scoring=scoring,
            random_state=RDS,
            verbose=0,
        )
        rand_cv.fit(X, y)
        self.model = rand_cv.best_estimator_

    def predict(self, eastings: list[float], northings: list[float]) -> pd.Series:
        X = np.c_[eastings, northings]
        predictions = self.model.predict(X)
        return self.format_output(X, self.le.inverse_transform(predictions))

    def predict_test(self) -> dict[str, numbers.Real]:
        """Extract performance metrics purely for model evaluation."""
        self.model.fit(self.X_train, self.y_train)
        return compute_metrics(
            y_true=self.y_test,
            y_hat=self.model.predict(self.X_test),
            y_pp=self.model.predict_proba(self.X_test),
            regression=False,
            average="macro",
            verbose=False,
        )

    def format_output(self, X: Any, predictions: Any) -> pd.Series:
        return pd.merge(
            pd.DataFrame(X, columns=["easting", "northing"]),
            pd.Series(predictions, name="localAuthority"),
            left_index=True,
            right_index=True,
        ).set_index(["easting", "northing"])["localAuthority"]
