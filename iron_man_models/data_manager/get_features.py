import json
import logging
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from iron_man_models.config import FEATURES_PATH


def get_feature_df(path: str = f"{FEATURES_PATH}/features.csv") -> pd.DataFrame:
    logging.info(f"Downloading from {path}")
    return apply_filters(pd.read_csv(path))


def get_feature_list(path: str = f"{FEATURES_PATH}/feature_list.json") -> list:
    with open(path, "r") as f:
        return json.loads(f.read())["features_list"]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Applying predetermined filters")

    logging.info(f"Rows before filters: {df.shape[0]}")

    mlflow.log_metric("df_rows", df.shape[0])
    # Define filters
    filter_conditions = (
        df["historical_sum(game_played-played_map=played_map)"] > 0
    ) & (df["historical_sum(game_played-played_map=played_map)_op"] > 0)

    # Apply filters
    filtered_df = df[filter_conditions]

    # Log the filter conditions to MLflow
    mlflow.log_param("number of games played on map filter", "> 0")

    # Define columns for dropna
    dropna_columns = [
        "simple_feature(overall_elo-shift=0)",
        "simple_feature(overall_elo-shift=0)_op",
    ]

    # Log columns to be checked for missing values
    mlflow.log_param("dropna_columns", dropna_columns)

    # Drop rows with NaN values in the specified columns
    filtered_df = filtered_df.dropna(subset=dropna_columns)
    # Log the shape of the filtered df
    logging.info(f"Rows after filters: {filtered_df.shape[0]}")
    mlflow.log_metric("filtered_df_rows", filtered_df.shape[0])

    return filtered_df


def custom_train_test_split(
    df: pd.DataFrame,
    oot_date_threshold: datetime,
    random_test_size: float,
    feature_list: list,
):
    train_index = df["match_date"].map(datetime.fromisoformat) <= oot_date_threshold
    remaining_df = df[train_index]
    oot_test_df = df[~train_index]

    feature_list = sorted(feature_list)

    train_df, random_test_df = train_test_split(
        remaining_df,
        test_size=random_test_size,
        random_state=42,
    )
    mlflow.log_metric("train_size", train_df.shape[0])
    mlflow.log_metric("random_test_size", random_test_df.shape[0])
    mlflow.log_metric("oot_test_size", oot_test_df.shape[0])

    logging.info(f"Train size: {train_df.shape[0]}")
    logging.info(f"Random test size: {random_test_df.shape[0]}")
    logging.info(f"OOT test size: {oot_test_df.shape[0]}")

    X_train = train_df[feature_list]
    y_train = train_df["won"]

    X_random_test = random_test_df[feature_list]
    y_random_test = random_test_df["won"]

    X_oot_test = oot_test_df[feature_list]
    y_oot_test = oot_test_df["won"]

    return (
        X_train,
        X_random_test,
        X_oot_test,
        y_train,
        y_random_test,
        y_oot_test,
    )
