import pandas as pd
from sklearn.svm import SVC

from iron_man_models.config import FEATURE_IMPORTANCE_PATH
from iron_man_models.models.base import BaseModel


class SVCModel(BaseModel):

    def build_model(self):
        """
        Build the SVC model with the provided parameters.
        """
        self.model = SVC(**self.params)

    def save_feature_importance(self, feature_list):
        importances = self.model.feature_importances_

        pd.DataFrame({"feature": feature_list, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(FEATURE_IMPORTANCE_PATH, index=False)
