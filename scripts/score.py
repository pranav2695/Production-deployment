import argparse

import mlflow

from fsds.score import score_model
from fsds.utils import log, setup_logger


def main(input_data_path, model_path, output_path, log_level, log_path, 
         no_console_log):
    # Use local folder-based MLflow tracking to ./mlartifacts
    mlflow.set_tracking_uri(uri="file:./mlartifacts")
    exp_name = "fsds-score"
    mlflow.set_experiment(exp_name)

    # Set up logging
    logger = setup_logger(
        log_level=log_level,
        log_path=log_path,
        console_log=not no_console_log,
    )

    # Score the model
    rmse = score_model(
        housing_path=input_data_path, model_path=model_path, output_path=output_path
    )

    # Log RMSE to MLflow
    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)

    logger.debug("Scoring the housing model completed")
    log("Scoring the housing model completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Housing Price Model")
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="datasets/housing",
        help="Path to load the training data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pkl",
        help="Path to load the trained model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/metrics/score.txt",
        help="Path to write the RMSE metric",
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
        args.input_data_path,
        args.model_path,
        args.output_path,
        log_level=args.log_level,
        log_path=args.log_path,
        no_console_log=args.no_console_log,
    )
