import logging
import mlflow
import numpy as np
# import pandas as pd
# from iron_man_models.config import FEATURE_IMPORTANCE_PATH
from iron_man_models.models.base import BaseModel
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseModel):
    param_grid: dict = {}

    def build_model(self):
        """
        Build the LGBMClassifier model with the provided parameters.
        """
        self.model = LogisticRegression(**self.params)

    def save_feature_importance(self, X):
        # The estimated coefficients will all be around 1:
        logging.info(self.model.coef_)

        # Those values, however, will show that the second parameter
        # is more influential
        logging.info(np.std(X, 0) * self.model.coef_)

    def fit(self, X_train, y_train):
        """
        Train the model using the training data.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        self.model.fit(X_train.fillna(0), y_train)
        mlflow.log_params(self.params)

    def predict_proba(self, X_test):
        """
        Generate predictions on the test data.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model not built. Use build_model() method first."
            )

        return self.model.predict_proba(X_test.fillna(0))
