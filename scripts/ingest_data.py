# Standard library imports
import argparse
import os

import mlflow

# Local imports
from fsds.ingestion import fetch_housing_data, load_housing_data
from fsds.utils import log, setup_logger


def main(output_path, log_level, log_path, no_console_log):
    # Use environment variable or fallback to local artifacts folder
    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlartifacts"))
    exp_name = "fsds-ingest-data"
    mlflow.set_experiment(exp_name)

    logger = setup_logger(
        log_level=log_level,
        log_path=log_path,
        console_log=not no_console_log,
    )

    logger.debug(f"Fetching and loading housing data into: {output_path}...")
    log("Fetching and loading housing data into:")

    fetch_housing_data(housing_path=output_path)
    df = load_housing_data(housing_path=output_path)

    logger.debug(f"Data loaded successfully with shape: {df.shape}")
    log("Data loaded successfully with shape:")

    with mlflow.start_run():
        # Log dataset metadata
        row_count = df.shape[0] if hasattr(df, "shape") else len(df)
        mlflow.log_param("dataset_row_count", row_count)
        mlflow.log_param("dataset_path", output_path)

        # Optional: log a sample as an artifact
        df_sample = df.head(50)
        sample_file = "sample_data.csv"
        df_sample.to_csv(sample_file, index=False)
        mlflow.log_artifact(sample_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Housing Dataset")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./datasets/housing",
        help="Output dir path where data will be stored (default: ./datasets/housing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-path", type=str, default=None, help="Optional path to a log file"
    )
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )

    args = parser.parse_args()
    main(
        args.output_path,
        log_level=args.log_level,
        log_path=args.log_path,
        no_console_log=args.no_console_log,
    )
