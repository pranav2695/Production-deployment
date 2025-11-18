import argparse

import mlflow

from fsds.train import train_and_save_model
from fsds.utils import log, setup_logger


def main(model_path, input_data_path, log_level, log_path, no_console_log):
    # Set MLflow tracking URI to a local folder
    mlflow.set_tracking_uri(uri="file:./mlartifacts")
    exp_name = "fsds-train"
    mlflow.set_experiment(exp_name)

    # Set up logging
    logger = setup_logger(
        log_level=log_level,
        log_path=log_path,
        console_log=not no_console_log,
    )

    logger.debug("Training the housing model...")

    # Train the model and get parameters and model object
    params, model = train_and_save_model(
        model_path=model_path, input_data_path=input_data_path
    )

    log("Completed the training of the model")

    # Log parameters and model to MLflow
    with mlflow.start_run():
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Housing Price Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="./datasets/housing",
        help="Path to read the input data from",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-path", type=str, default=None, 
        help="Optional path to a log file"
    )
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()
    main(
        args.model_path,
        args.input_data_path,
        log_level=args.log_level,
        log_path=args.log_path,
        no_console_log=args.no_console_log,
    )
