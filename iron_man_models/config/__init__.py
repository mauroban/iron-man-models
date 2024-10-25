import os


def safe_path_join(*args):
    """Join parts of a URL or file path, ensuring there's exactly one separator."""
    # Remove any trailing or leading slashes from parts and join with a single slash
    return "/".join(part.strip("/") for part in args if part)


# Configured paths
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", "data/processed")
UNPROCESSED_FOLDER = os.getenv("UNPROCESSED_FOLDER", "data/unprocessed")
ERROR_FOLDER = os.getenv("ERROR_FOLDER", "data/errors")
FEATURES_PATH = os.getenv("FEATURES_PATH", "../iron-man-features/data")
FEATURE_IMPORTANCE_PATH = os.getenv(
    "FEATURE_IMPORTANCE_PATH", "results/feature_importance.csv"
)
MATCHES_TO_PREDICT_PATH = os.getenv(
    "MATCHES_TO_PREDICT_PATH", "../iron-man-features/data/matches_to_predict.csv"
)
