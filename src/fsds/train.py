import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fsds.ingestion import fetch_housing_data, load_housing_data


def prepare_data(data):
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_idx].drop("income_cat", axis=1)
        strat_test_set = data.loc[test_idx].drop("income_cat", axis=1)

    return strat_train_set, strat_test_set


def create_pipeline(num_features, cat_features):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    return full_pipeline


def train_and_save_model(
    model_path="models/best_model.pkl", input_data_path="./datasets/housing"
):
    fetch_housing_data()
    housing = load_housing_data(input_data_path)
    train_set, _ = prepare_data(housing)

    housing_labels = train_set["median_house_value"].copy()
    housing_features = train_set.drop("median_house_value", axis=1)

    housing_features["rooms_per_household"] = (
        housing_features["total_rooms"] / housing_features["households"]
    )
    housing_features["bedrooms_per_room"] = (
        housing_features["total_bedrooms"] / housing_features["total_rooms"]
    )
    housing_features["population_per_household"] = (
        housing_features["population"] / housing_features["households"]
    )

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = create_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    param_grid = {
        "n_estimators": [30, 50],
        "max_features": [6, 8],
    }

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((grid_search.best_estimator_, pipeline), model_path)
    print(f"Model saved to {model_path}")
    return grid_search.best_params_, grid_search.best_estimator_
