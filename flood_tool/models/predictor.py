import numbers
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """Parent class framework for all predictors based on an OOP style."""

    def __init__(
        self,
        training_data: pd.DataFrame,
        model_alias: str,
        scoring: Any,
        tune_hyperparameters: bool = False,
        n_iter: int = 10,
    ):
        # initialise class attributes
        self.training_data = training_data
        self.model_alias = model_alias
        self.scoring = scoring
        self.test_score = None

        self.X, self.X_train, self.X_test, self.y, self.y_train, self.y_test = [None] * 6

        self.model = None
        self.le = None

        self.prepare_data()
        self.build_pipeline()

        if tune_hyperparameters:  # tune on train data
            self.tune_hyperparameters(
                X=self.X_train, y=self.y_train, scoring=scoring, n_iter=n_iter
            )

        self.predict_test()  # produce test results

        self.model.fit(self.X, self.y)  # fit using all data
        logger.info("Model successfully fit.")

    @abstractmethod
    def prepare_data(self) -> None:
        """Should return X and y before splitting"""
        pass

    @abstractmethod
    def build_pipeline(self) -> None:
        pass

    @abstractmethod
    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, scoring: Any, n_iter: int
    ) -> None:
        pass

    @abstractmethod
    def predict_test(self) -> dict[str, numbers.Real]:
        pass

    @abstractmethod
    def format_output(self, X: Any, predictions: Any, **kwargs) -> pd.Series:
        pass
