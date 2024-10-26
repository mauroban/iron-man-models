import mlflow
import pandas as pd
from xgboost import XGBClassifier

from iron_man_models.config import FEATURE_IMPORTANCE_PATH
from iron_man_models.models.base import BaseModel


class XGBoostModel(BaseModel):
    param_grid = {
        'n_estimators': [80, 100, 200, 300, 500],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5, 7, 10, 15],
        'gamma': [0, 0.1, 0.3, 0.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1]
    }

    def build_model(self):
        """
        Build the XGBClassifier model with the provided parameters.
        """
        self.model = XGBClassifier(**self.params)

    def save_feature_importance(self, feature_list):
        importances = self.model.feature_importances_

        pd.DataFrame({"feature": feature_list, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        mlflow.log_artifact(FEATURE_IMPORTANCE_PATH)
