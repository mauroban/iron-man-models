import logging

import mlflow
from sklearn.linear_model import LogisticRegression

# import pandas as pd
# from iron_man_models.config import FEATURE_IMPORTANCE_PATH
from iron_man_models.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    param_grid: dict = {}

    def build_model(self):
        """
        Build the LGBMClassifier model with the provided parameters.
        """
        if not self.params:
            self.params = {}
        self.model = LogisticRegression(**self.params)

    def save_feature_importance(self, feature_list):
        logging.info(self.model.coef_)
        # logging.info(np.std(feature_list, 0) * self.model.coef_)

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
