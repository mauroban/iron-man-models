from datetime import datetime

import mlflow

from iron_man_models.data_manager.get_features import (
    custom_train_test_split,
    get_feature_df,
    get_feature_list,
)
from iron_man_models.models.lgbm_model import LGBMModel
from iron_man_models.models.logistic_regression_model import LogisticRegressionModel
from iron_man_models.models.svc_model import SVCModel
from iron_man_models.models.xgboost_model import XGBoostModel


_ESTIMATORS = {
    "LGBM": LGBMModel,
    "LogisticRegression": LogisticRegressionModel,
    "SVC": SVCModel,
    "XGBoost": XGBoostModel,
}


def train_flow(algorithm: str, experiment_name: str = "iron-man", run_name: str = None):
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("algorithm", algorithm)
        data = get_feature_df()
        features = get_feature_list()

        (
            X_train,
            X_random_test,
            X_oot_test,
            y_train,
            y_random_test,
            y_oot_test,
        ) = custom_train_test_split(
            df=data,
            oot_date_threshold=datetime(2024, 10, 15),
            random_test_size=0.06,
            feature_list=features,
        )

        estimator = _ESTIMATORS.get(algorithm)()
        estimator.build_model()
        estimator.tune_hyperparameters(
            X_train,
            y_train,
            n_iter=10,
            cv=5,
        )

        estimator.fit(X_train, y_train)

        estimator.evaluate(
            X_train=X_train,
            y_train=y_train,
            X_random_test=X_random_test,
            y_random_test=y_random_test,
            X_oot_test=X_oot_test,
            y_oot_test=y_oot_test,
        )

        estimator.save_feature_importance(feature_list=features)
        estimator.save_model("results/model.pkl")


if __name__ == "__main__":
    import importlib
    import logging

    importlib.reload(logging)

    logging.basicConfig(
        format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        encoding="utf-8",
    )
    train_flow(
        algorithm="XGBoost",
        run_name="remove_slow_elo",
    )
