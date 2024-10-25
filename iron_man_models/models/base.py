import logging
from abc import ABC, abstractmethod

import joblib
import mlflow
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV


class BaseModel(ABC):
    param_grid: dict = {}

    def __init__(self, params={}, model: BaseEstimator = None):
        """
        Initialize the model with parameters.
        """
        self.model = model
        self.params = params

    @abstractmethod
    def build_model(self):
        """
        Abstract method to build the model.
        This should be implemented in child classes.
        """
        pass

    def fit(self, X_train, y_train):
        """
        Train the model using the training data.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        self.model.fit(X_train, y_train)
        mlflow.log_params(self.params)

    def set_param_grid(self, **kwargs):
        logging.info(kwargs)
        self.param_grid = self.param_grid

    def tune_hyperparameters(
        self, X_train, y_train, n_iter=10, cv=5, scoring="neg_log_loss"
    ):
        """
        Tune the model's hyperparameters using GridSearchCV.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        if self.param_grid:
            grid_search = RandomizedSearchCV(
                self.model, self.param_grid, n_iter=n_iter, cv=cv, scoring=scoring
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.params = grid_search.best_params_

            mlflow.log_params(self.params)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)

            return grid_search.best_estimator_, grid_search.best_params_

    def predict_proba(self, X_test):
        """
        Generate predictions on the test data.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        return self.model.predict_proba(X_test)

    def evaluate(
        self,
        X_train,
        y_train,
        X_random_test,
        y_random_test,
        X_oot_test,
        y_oot_test,
        scoring_func=log_loss,
    ):
        """
        Evaluate the model using a custom scoring function.
        """
        y_pred = self.predict_proba(X_train)
        score = scoring_func(y_train, y_pred)
        logging.info(f"biased_train log loss: {score}")
        mlflow.log_metric("biased_train log loss", score)

        y_pred = self.predict_proba(X_random_test)
        score = scoring_func(y_random_test, y_pred)
        logging.info(f"random_test log loss: {score}")
        mlflow.log_metric("random_test log loss", score)

        y_pred = self.predict_proba(X_oot_test)
        score = scoring_func(y_oot_test, y_pred)
        logging.info(f"oot_test log loss: {score}")
        mlflow.log_metric("oot_test log loss", score)

    def save_model(self, filepath: str):
        """
        Save the pure model (not the custom class) to a file using joblib.
        This allows the model to be reused in other projects without this class.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        joblib.dump(self.model, filepath)
        mlflow.log_artifact(filepath)
        logging.info(f"Model saved to {filepath}")

    def save_feature_importance(self):
        raise NotImplementedError(
            "Function was not implemented on class"
        )

    @staticmethod
    def load_model(filepath: str):
        """
        Load a saved model from a file.
        """
        return joblib.load(filepath)
