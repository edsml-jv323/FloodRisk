import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
import logging
import numbers
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

from flood_tool.models.constant import TEST_SIZE, RDS
from flood_tool.models.predictor import Predictor
from utils.utils import binning, compute_metrics

logger = logging.getLogger(__name__)


class MedianPricePredictor(Predictor):
    def __init__(
        self,
        training_data: pd.DataFrame,
        model_alias: str,
        scoring: Any,
        tune_hyperparameters: bool = False,
        n_iter: int = 25,
    ):
        self.n_iter = n_iter
        self.num_cols = ["easting", "northing", "elevation"]
        self.cat_cols = ["soilType"]

        super().__init__(training_data, model_alias, scoring, tune_hyperparameters, n_iter)

    def prepare_data(self) -> None:
        data = self.training_data
        data.medianPrice = np.nan_to_num(data.medianPrice, nan=data.medianPrice.mean())

        self.y = np.log(data["medianPrice"] + 1e-10)
        self.X = data.drop(columns=["medianPrice"])[self.num_cols + self.cat_cols]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            random_state=RDS,
        )

    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the dataset."""
        Q1 = data["medianPrice"].quantile(0.25)
        Q3 = data["medianPrice"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[
            (data["medianPrice"] >= lower_bound) & (data["medianPrice"] <= upper_bound)
        ]
        return data

    def build_pipeline(self) -> None:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols),
                ("cat", cat_pipe, self.cat_cols),
            ]
        )

        match self.model_alias:
            case "linear_regression":
                model = LinearRegression()
            case "knn_regressor":
                model = KNeighborsRegressor()
            case _:
                raise ValueError("Unknown model alias.")

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, scoring: Any, n_iter: int
    ) -> None:
        raise NotImplementedError
        logger.info("Tuning hyperparameters...")

        param_distribution = {
            "model__fit_intercept": [True, False],
            "model__copy_X": [True, False],
            "model__alpha": stats.uniform(0, 1),
            "model__l1_ratio": stats.uniform(0, 1),
        }

        rand_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distribution,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=5, shuffle=False).split(
                self.X_train, y=binning(self.y_train)
            ),
            n_iter=n_iter,
            n_jobs=-1,
            random_state=RDS,
        )

        rand_cv.fit(X, y)
        self.model = rand_cv.best_estimator_

    def predict_test(self) -> dict[str, numbers.Real]:
        self.model.fit(self.X_train, self.y_train)
        y_hat = np.exp(self.model.predict(self.X_test))
        scores = compute_metrics(np.exp(self.y_test), y_hat, regression=True)
        return scores

    def predict_from_postcode(self, X: pd.DataFrame, index: str) -> pd.Series:
        predictions = self.model.predict(X)
        return self.format_output(X, np.exp(predictions), index=index)

    def format_output(self, X: Any, predictions: Any, **kwargs) -> pd.Series:
        index = kwargs.get("index", None)
        if index is None:
            raise ValueError("Index must be specified")
        return pd.Series(predictions.astype(int), index=X[index], name="predictions")
