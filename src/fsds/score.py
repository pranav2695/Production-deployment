import os

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from fsds.ingestion import load_housing_data
from fsds.train import prepare_data


def score_model(
    housing_path,
    model_path="models/best_model.pkl",
    output_path="models/metrics/score.txt",
):
    housing = load_housing_data(housing_path)
    _, test_set = prepare_data(housing)

    test_labels = test_set["median_house_value"].copy()
    test_features = test_set.drop("median_house_value", axis=1)

    test_features["rooms_per_household"] = (
        test_features["total_rooms"] / test_features["households"]
    )
    test_features["bedrooms_per_room"] = (
        test_features["total_bedrooms"] / test_features["total_rooms"]
    )
    test_features["population_per_household"] = (
        test_features["population"] / test_features["households"]
    )

    model, pipeline = joblib.load(model_path)
    test_prepared = pipeline.transform(test_features)
    predictions = model.predict(test_prepared)

    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
    print(f"RMSE on test data: {rmse:.2f}")
    return rmse
