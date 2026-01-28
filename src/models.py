"""Machine learning models for energy consumption analysis."""

from typing import Dict, List, Tuple

from numpy import ndarray, sqrt
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
from statsmodels.regression.linear_model import RegressionResultsWrapper


def scale_features(
    train_features: DataFrame, test_features: DataFrame
) -> Tuple[ndarray, ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.

    Args:
        train_features: Training features DataFrame
        test_features: Test features DataFrame

    Returns:
        Tuple of (scaled_train_features, scaled_test_features, scaler)
    """
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)

    return scaled_train_features, scaled_test_features, scaler


def train_ols_regression(
    train_features: ndarray, train_target: Series
) -> RegressionResultsWrapper:
    """
    Train OLS regression model using statsmodels.

    Args:
        train_features: Scaled training features
        train_target: Training target values

    Returns:
        Fitted OLS model
    """
    # Add constant term for intercept
    features_with_constant = add_constant(train_features)

    # Fit OLS model
    model = OLS(train_target, features_with_constant).fit()

    return model


def predict_ols(model: RegressionResultsWrapper, test_features: ndarray) -> ndarray:
    """
    Make predictions using fitted OLS model.

    Args:
        model: Fitted OLS model
        test_features: Scaled test features

    Returns:
        Predicted values
    """
    # Add constant term for intercept
    features_with_constant = add_constant(test_features)

    predictions = model.predict(features_with_constant)

    # Return as numpy array (predictions may already be ndarray)
    return predictions if isinstance(predictions, ndarray) else predictions.values


def evaluate_regression(
    true_values: ndarray, predicted_values: ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Args:
        true_values: True target values
        predicted_values: Predicted target values

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "MAE": mean_absolute_error(true_values, predicted_values),
        "MSE": mean_squared_error(true_values, predicted_values),
        "RMSE": sqrt(mean_squared_error(true_values, predicted_values)),
        "R2": r2_score(true_values, predicted_values),
    }

    return metrics


def determine_optimal_clusters(
    data: ndarray, max_clusters: int = 10
) -> Tuple[List[int], List[float]]:
    """
    Determine optimal number of clusters using elbow method.

    Args:
        data: Scaled features array
        max_clusters: Maximum number of clusters to test

    Returns:
        Tuple of (k_values, inertias)
    """
    k_values = list(range(1, max_clusters + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    return k_values, inertias


def train_kmeans_clustering(data: ndarray, n_clusters: int) -> Tuple[KMeans, ndarray]:
    """
    Train K-means clustering model.

    Args:
        data: Scaled features array
        n_clusters: Number of clusters

    Returns:
        Tuple of (fitted_model, cluster_labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    return kmeans, cluster_labels


def analyze_clusters(dataframe: DataFrame, cluster_labels: ndarray) -> DataFrame:
    """
    Analyze cluster characteristics by calculating mean values for numeric features.

    Args:
        dataframe: Original DataFrame with features
        cluster_labels: Cluster assignments for each sample

    Returns:
        DataFrame with cluster statistics (mean values for numeric columns)
    """
    # Add cluster labels to dataframe
    analysis_dataframe = dataframe.copy()
    analysis_dataframe["Cluster"] = cluster_labels

    # Calculate mean values for numeric columns only
    cluster_statistics = analysis_dataframe.groupby("Cluster").mean(numeric_only=True)

    # Add cluster sizes
    cluster_statistics["Count"] = analysis_dataframe.groupby("Cluster").size()

    return cluster_statistics


def calculate_vif(features: DataFrame) -> DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity check.

    Args:
        features: Features DataFrame

    Returns:
        DataFrame with VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = DataFrame()
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(features.values, i)
        for i in range(len(features.columns))
    ]

    return vif_data
