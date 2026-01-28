"""Data loading and preprocessing functions for energy consumption analysis."""

from typing import Tuple

from pandas import DataFrame, Series, get_dummies, read_csv


def load_energy_data(train_path: str, test_path: str) -> Tuple[DataFrame, DataFrame]:
    """
    Load training and test energy consumption datasets.
    Automatically cleans column names by replacing spaces with underscores.

    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file

    Returns:
        Tuple of (train_dataframe, test_dataframe)
    """
    train_dataframe = read_csv(train_path)
    test_dataframe = read_csv(test_path)

    # Clean column names: replace spaces with underscores
    train_dataframe.columns = train_dataframe.columns.str.replace(" ", "_")
    test_dataframe.columns = test_dataframe.columns.str.replace(" ", "_")

    return train_dataframe, test_dataframe


def get_data_info(dataframe: DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.

    Args:
        dataframe: Input DataFrame

    Returns:
        Dictionary with dataset information
    """
    info = {
        "shape": dataframe.shape,
        "columns": dataframe.columns.tolist(),
        "dtypes": dataframe.dtypes.to_dict(),
        "missing_values": dataframe.isnull().sum().to_dict(),
        "memory_usage": dataframe.memory_usage(deep=True).sum() / 1024**2,  # MB
    }

    return info


def prepare_features_target(
    dataframe: DataFrame, target_col: str = "Energy_Consumption"
) -> Tuple[DataFrame, Series]:
    """
    Split dataset into features and target variable.

    Args:
        dataframe: Input DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (features, target) where features is DataFrame and target is Series
    """
    features = dataframe.drop(columns=[target_col])
    target = dataframe[target_col]

    return features, target


def encode_categorical_features(features: DataFrame) -> DataFrame:
    """
    Encode categorical features using one-hot encoding.
    Drops first category to avoid multicollinearity (dummy variable trap).

    Args:
        features: Features DataFrame

    Returns:
        DataFrame with encoded categorical features
    """
    categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()

    if not categorical_cols:
        return features

    # One-hot encode and drop first category
    features_encoded = get_dummies(
        features, columns=categorical_cols, drop_first=True, dtype=int
    )

    return features_encoded


def get_numeric_features(dataframe: DataFrame) -> DataFrame:
    """
    Extract only numeric features from DataFrame.

    Args:
        dataframe: Input DataFrame

    Returns:
        DataFrame with only numeric columns
    """
    return dataframe.select_dtypes(include=["number"])


def get_categorical_features(dataframe: DataFrame) -> DataFrame:
    """
    Extract only categorical features from DataFrame.

    Args:
        dataframe: Input DataFrame

    Returns:
        DataFrame with only categorical columns
    """
    return dataframe.select_dtypes(include=["object"])
