import argparse

import mlflow

from fsds.ingestion import fetch_housing_data, load_housing_data
from fsds.score import score_model
from fsds.train import train_and_save_model


def main(use_mlflow):
    fetch_housing_data(housing_path="./datasets/housing")
    df = load_housing_data(housing_path="./datasets/housing")
    row_count = df.count()
    # Training the model
    params, model = train_and_save_model(
        model_path="models/best_model.pkl", 
        input_data_path="./datasets/housing"
    )
    # Scoring of the model
    rmse = score_model(
        housing_path="./datasets/housing",
        model_path="models/best_model.pkl",
        output_path="models/metrics/score.txt",
    )
    if use_mlflow:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5050")
        exp_name = "housing-price-pipeline"
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="Full Pipeline Run"):
            # mlflow run to log the ingest data
            with mlflow.start_run(run_name="Ingest Data", nested=True):
                mlflow.log_param("dataset_row_count", row_count)
                # Save path or version of dataset
                mlflow.log_param("dataset_path", "./datasets/housing")
                # log a sample as an artifact
                df_sample = df.head(50)
                df_sample.to_csv("sample_data.csv", index=False)
                mlflow.log_artifact("sample_data.csv")
            # mlflow run to log the training of the model
            with mlflow.start_run(run_name="Train Model", nested=True):
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                mlflow.sklearn.log_model(model, "model")
            with mlflow.start_run(run_name="Score Model", nested=True):
                mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Locally we can do the tracking using mlflow"
    )
    parser.add_argument(
        "--use_mlflow",
        type=str,
        default=False,
        help="use mlflow when we are running locally",
    )
    args = parser.parse_args()
    main(args.use_mlflow)
