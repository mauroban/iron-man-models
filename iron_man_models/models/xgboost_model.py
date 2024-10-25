import pandas as pd
from xgboost import XGBClassifier

from iron_man_models.config import FEATURE_IMPORTANCE_PATH
from iron_man_models.models.base import BaseModel


class XgboostModel(BaseModel):
    param_grid: dict = {
        "num_leaves": [4, 5, 6, 7, 8],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.007, 0.01, 0.012, 0.015],
        "n_estimators": [1200, 2000, 3000, 5000],
        "min_split_gain": [0.0],
        "min_child_weight": [0.001],
        "min_child_samples": [20],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "reg_alpha": [0.0],
        "reg_lambda": [0.0],
        # 'scale_pos_weight': np.arange(1, 7),
        "boosting_type": ["dart"],
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
