import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from xgboost import XGBRegressor

from imblearn.pipeline import Pipeline as ImbPipeline

from flood_tool.models.constant import TEST_SIZE, RDS
from flood_tool.models.predictor import Predictor
from scoring.scores import SCORES

logger = logging.getLogger(__name__)


class FloodPredictor(Predictor):
    def __init__(
        self,
        training_data: pd.DataFrame,
        model_alias: str,
        scoring: Any,
        tune_hyperparameters: bool = False,
        n_iter: int = 10,
    ):
        self.num_cols = ["easting", "northing", "elevation"]
        self.cat_cols = ["soilType"]
        super().__init__(training_data, model_alias, scoring, tune_hyperparameters, n_iter)

    def prepare_data(self) -> None:
        self.X = self.training_data.drop(columns="riskLabel")[
            self.num_cols + self.cat_cols
        ].copy()
        self.y = self.training_data.riskLabel.copy()

        if self.model_alias != "rf_class_downsampling":
            # this is essentially our label encoding, don't forget to reverse!
            self.y -= 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            random_state=RDS,
            stratify=self.y,
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
            case "xgb_regressor":
                self.model = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        (
                            "model",
                            XGBRegressor(
                                learning_rate=0.775,
                                max_depth=41,
                                n_estimators=917,
                                random_state=RDS,
                            ),
                        ),
                    ]
                )
            case "rf_smote" | "rf_class_downsampling":
                num_pipe = Pipeline(
                    [("imputer", SimpleImputer()), ("scaler", StandardScaler())]
                )

                cat_pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                )

                preprocessor = ColumnTransformer(
                    [
                        (
                            "num",
                            num_pipe,
                            self.num_cols,
                        ),
                        (
                            "cat",
                            cat_pipe,
                            self.cat_cols,
                        ),
                    ]
                )

                # final pipeline with SMOTE and the classifier
                self.model = ImbPipeline(
                    [
                        ("preprocessor", preprocessor),
                        # ("smote", SMOTE(random_state=RDS)),
                        (
                            "classifier",
                            RandomForestClassifier(n_jobs=-1, random_state=RDS),
                        ),
                    ]
                )
            case _:
                raise ValueError("Unknown model alias.")

    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, scoring: Any, n_iter: int
    ) -> None:
        logger.info("Tuning hyperparameters...")

        match self.model_alias:
            case "xgb_regressor":
                param_distribution = {
                    "preprocessor__num__scaler": [StandardScaler(), MinMaxScaler()],
                    "model__learning_rate": stats.uniform(0.01, 0.9),
                    "model__n_estimators": stats.randint(500, 1000),
                    "model__max_depth": stats.randint(30, 60),
                }
            case "rf_smote":
                param_distribution = {}
            case "rf_class_downsampling":
                param_distribution = {}
            case _:
                raise ValueError("Unknown model alias.")

        rand_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distribution,
            scoring=make_scorer(scoring, greater_is_better=True),
            cv=StratifiedKFold(n_splits=5, shuffle=False).split(
                self.X_train, y=self.y_train
            ),
            n_iter=n_iter,
            n_jobs=-1,
            random_state=RDS,
        )

        rand_cv.fit(X, y)
        self.model = rand_cv.best_estimator_

    def merge_categories(self, y) -> Any:
        return y.replace({2: 1, 3: 1, 4: 7, 5: 7, 6: 7, 8: 9, 10: 9})

    def predict_test(self) -> None:
        if self.model_alias == "rf_class_downsampling":
            y_train_mapped = self.merge_categories(self.y_train)
            probabilities = self.calculate_probabilities(self.y_train)
            self.model.fit(self.X_train, y_train_mapped)
            predictions_mappped = self.model.predict(self.X_test)
            predictions = self.expand_predictions(predictions_mappped, probabilities)
            self.test_score = {
                "flood_score": self.classifier_scorer(self.y_test - 1, predictions - 1)
            }  # to fit with scoring
        elif self.model_alias == "rf_smote":
            self.model.fit(self.X_train, self.y_train)
            y_hat = self.model.predict(self.X_test)
            self.test_score = self.classifier_scorer(self.y_test, y_hat)
        else:
            self.model.fit(self.X_train, self.y_train)
            y_hat = self.model.predict(self.X_test)
            y_hat = np.round(y_hat)
            y_hat = np.clip(y_hat, 0, y_hat.max())
            self.test_score = {"flood_score": self.classifier_scorer(self.y_test, y_hat)}

    def predict_flood_risk(
        self, X: pd.DataFrame, index: tuple[str, str] | str
    ) -> pd.Series:
        """Index should be the list of values specified
        unless it's postcode, in which case it can be a string."""
        self.model.fit(self.X, self.y)

        match self.model_alias:
            case "rf_class_downsampling":
                predictions_mappped = self.model.predict(X)
                probabilities = self.calculate_probabilities(self.y)
                predictions = self.expand_predictions(predictions_mappped, probabilities)
            case "rf_smote":
                predictions = self.model.predict(X)
                predictions += 1
            case "xgb_regressor":
                predictions = self.model.predict(X)
                predictions = np.round(predictions)
                predictions = np.clip(predictions, 0, predictions.max()) + 1
            case _:
                raise ValueError("Model alias not known.")
        return self.format_output(X, predictions, index=index)

    def format_output(self, X: Any, predictions: Any, **kwargs) -> pd.Series:
        index = kwargs.get("index", None)
        if index is None:
            raise ValueError("Index must be specified")

        if isinstance(index, tuple):
            X.drop(columns=list(index), inplace=True)
            X.rename(columns={"x": index[0], "y": index[1]}, inplace=True)
            multi_index = pd.MultiIndex.from_arrays([X[index[0]], X[index[1]]])
            return pd.Series(predictions.astype(int), index=multi_index, name="predictions")
        else:
            return pd.Series(predictions.astype(int), index=X[index], name="predictions")

    @staticmethod
    def classifier_scorer(y, predictions):
        """Should take discrete outputs 0 indexed"""
        return sum(
            SCORES[pred, true] for pred, true in zip(predictions.astype(int), y.astype(int))
        )

    @staticmethod
    def calculate_probabilities(y_train: pd.Series) -> dict:
        merged_to_original = {1: [1, 2, 3], 7: [4, 5, 6, 7], 9: [8, 9, 10]}

        probabilities = {}

        for merged_cat, original_cats in merged_to_original.items():
            subset = y_train[y_train.isin(original_cats)]
            subset_counts = subset.value_counts()
            total_count = subset_counts.sum()
            probabilities[merged_cat] = {
                cat: count / total_count for cat, count in subset_counts.items()
            }

        return probabilities

    @staticmethod
    def expand_predictions(predictions, probabilities) -> np.ndarray:
        expanded_predictions = []

        for pred in predictions:
            if pred in probabilities:  # check if the prediction is in the merged categories
                original_cats = list(probabilities[pred].keys())
                probs = list(probabilities[pred].values())
                expanded_pred = np.random.choice(original_cats, p=probs)
                expanded_predictions.append(expanded_pred)
            else:
                # if the prediction is not in the merged categories, keep it as is
                expanded_predictions.append(pred)

        return np.array(expanded_predictions)
