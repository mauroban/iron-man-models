import logging
import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from iron_man_models.models.base import BaseModel
# from scipy.stats import reciprocal, uniform


class SVCModel(BaseModel):
    # param_grid = {
    #     "gamma": reciprocal(0.001, 0.1),
    #     "C": uniform(1, 10),
    # }

    def build_model(self):
        """
        Build the SVC model with the provided parameters.
        """
        if not self.params:
            self.params = {"probability": True}
        self.model = SVC(**self.params)

    def save_feature_importance(self, feature_list):
        logging.info("There is no feature importance to SVC")
        return

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
            grid_search.fit(X_train.fillna(0), y_train)
            mlflow.log_param("CV n_iter", n_iter)
            mlflow.log_param("CV folds", cv)

            self.model = grid_search.best_estimator_
            self.params = grid_search.best_params_

            mlflow.log_params(self.params)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)

            return grid_search.best_estimator_, grid_search.best_params_
