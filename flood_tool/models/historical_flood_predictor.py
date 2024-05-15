import logging
from typing import Any

import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as imblearnPipeline
from xgboost import XGBClassifier

from flood_tool.models.constant import TEST_SIZE, RDS
from flood_tool.models.predictor import Predictor
from utils.utils import compute_metrics

logger = logging.getLogger(__name__)


class HistoricalFloodPredictor(Predictor):
    def __init__(
        self,
        training_data: pd.DataFrame,
        model_alias: str,
        scoring: Any,
        tune_hyperparameters: bool = False,
        n_iter: int = 10,
    ):
        self.num_cols = [
            "latitude",
            "longitude",
            "elevation",
            "typical_average_rainfall_per_hour",
            "wet_average_rainfall_per_hour",
        ]
        self.cat_cols = ["soilType", "postcodeSector"]

        super().__init__(training_data, model_alias, scoring, tune_hyperparameters, n_iter)

    def prepare_data(self) -> None:
        data = self.training_data

        self.X = data.drop(columns="historicallyFlooded")[self.num_cols + self.cat_cols]
        self.y = data.historicallyFlooded.astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RDS, stratify=self.y
        )

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
            case "baseline_linear":
                self.model = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        (
                            "model",
                            LogisticRegression(
                                class_weight="balanced",
                                multi_class="ovr",
                                max_iter=1000,
                                random_state=RDS,
                            ),
                        ),
                    ]
                )
            case "rf_classifier":
                self.model = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        (
                            "model",
                            RandomForestClassifier(
                                class_weight="balanced", random_state=42
                            ),
                        ),
                    ]
                )
            case "rf_smote":
                self.model = imblearnPipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("oversample", SMOTE(random_state=42)),
                        (
                            "model",
                            RandomForestClassifier(
                                class_weight="balanced_subsample",
                                random_state=42,
                                max_depth=33,
                                min_samples_leaf=3,
                                n_estimators=761,
                            ),
                        ),
                    ]
                )
            case "xgb_classifier":
                self.model = imblearnPipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("oversample", ADASYN(random_state=42)),
                        (
                            "model",
                            XGBClassifier(
                                scale_pos_weight=1,
                                use_label_encoder=False,
                                eval_metric="logloss",
                                random_state=42,
                            ),
                        ),
                    ]
                )
            case _:
                raise ValueError("Not a known model alias.")

    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, scoring: Any, n_iter: int
    ) -> None:
        logger.info("Tuning hyperparameters...")
        match self.model_alias:
            case "rf_classifier":
                param_distribution = {
                    "model__class_weight": ["balanced", "balanced_subsample"],
                    "model__max_depth": stats.randint(10, 100),
                    "model__n_estimators": stats.randint(100, 1000),
                    "model__min_samples_leaf": stats.randint(1, 20),
                }
            case _:
                raise ValueError("Unknown model alias.")

        rand_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distribution,
            scoring=make_scorer(scoring, greater_is_better=True),
            cv=StratifiedKFold(n_splits=5).split(self.X_train, y=self.y_train),
            n_iter=n_iter,
            n_jobs=-1,
            random_state=RDS,
        )

        rand_cv.fit(X, y)
        self.model = rand_cv.best_estimator_

    def predict_test(self) -> None:
        self.model.fit(self.X_train, self.y_train)
        y_hat = self.model.predict(self.X_test)
        self.test_score = compute_metrics(
            self.y_test, y_hat, regression=False, verbose=False
        )

    def predict_historical_flood(
        self, X: pd.DataFrame, index: tuple[str, str] | str
    ) -> pd.Series:
        """
        Predict historical flooding based on unseen samples.
        Index should be the list of values specified unless it's postcode,
        in which case it can be a string.
        """
        predictions = self.model.predict(X)
        return self.format_output(X, predictions, index=index)

    def format_output(self, X: Any, predictions: Any, **kwargs) -> pd.Series:
        index = kwargs.get("index", None)
        if index is None:
            raise ValueError("Index must be specified")

        return pd.Series(predictions.astype(int), index=X[index], name="predictions")
