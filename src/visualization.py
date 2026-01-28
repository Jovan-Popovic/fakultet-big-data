"""Visualization functions for energy consumption analysis."""

from matplotlib.pyplot import figure, scatter, plot, axhline, title, xlabel, ylabel, legend, grid, xticks, savefig, show, text, tight_layout, colorbar
from seaborn import set_style, boxplot, heatmap, pairplot
from pandas import DataFrame, Series
from numpy import ndarray, polyfit, poly1d
from scipy import stats
from typing import Optional, List


# Set seaborn style for professional appearance
set_style("whitegrid")


def plot_histogram_with_kde(data: Series, plot_title: str, x_label: str,
                            save_path: Optional[str] = None) -> None:
    """
    Plot histogram with KDE curve.

    Args:
        data: Data to plot
        plot_title: Plot title
        x_label: X-axis label
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))

    # Use matplotlib hist directly
    import matplotlib.pyplot as plt
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)

    # Add KDE curve
    data.plot(kind='kde', color='red', linewidth=2)

    title(plot_title, fontsize=14, fontweight='bold')
    xlabel(x_label, fontsize=12)
    ylabel('Density', fontsize=12)
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_bar_chart(data: Series, plot_title: str, x_label: str, y_label: str,
                   save_path: Optional[str] = None) -> None:
    """
    Plot bar chart for categorical data.

    Args:
        data: Data to plot (value counts)
        plot_title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))
    value_counts = data.value_counts()
    value_counts.plot(kind='bar', color='steelblue', edgecolor='black')

    title(plot_title, fontsize=14, fontweight='bold')
    xlabel(x_label, fontsize=12)
    ylabel(y_label, fontsize=12)
    xticks(rotation=45, ha='right')
    grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for index, value in enumerate(value_counts):
        text(index, value + max(value_counts) * 0.01, str(value),
             ha='center', va='bottom', fontweight='bold')

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_scatter_with_regression(x_data: Series, y_data: Series, plot_title: str,
                                 x_label: str, y_label: str,
                                 save_path: Optional[str] = None) -> None:
    """
    Plot scatter plot with regression line.

    Args:
        x_data: X-axis data
        y_data: Y-axis data
        plot_title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))

    # Scatter plot
    scatter(x_data, y_data, alpha=0.6, edgecolors='black', linewidths=0.5)

    # Add regression line
    coefficients = polyfit(x_data, y_data, 1)
    polynomial = poly1d(coefficients)
    plot(x_data, polynomial(x_data), "r-", linewidth=2,
         label=f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

    title(plot_title, fontsize=14, fontweight='bold')
    xlabel(x_label, fontsize=12)
    ylabel(y_label, fontsize=12)
    legend()
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_boxplot_by_category(data: DataFrame, category_column: str, value_column: str,
                             plot_title: str, x_label: str, y_label: str,
                             save_path: Optional[str] = None) -> None:
    """
    Plot box plot grouped by categorical variable.

    Args:
        data: Input DataFrame
        category_column: Categorical column name for grouping
        value_column: Numeric column name for values
        plot_title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))

    boxplot(data=data, x=category_column, y=value_column, palette='Set2')

    title(plot_title, fontsize=14, fontweight='bold')
    xlabel(x_label, fontsize=12)
    ylabel(y_label, fontsize=12)
    xticks(rotation=45, ha='right')
    grid(True, alpha=0.3, axis='y')

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_correlation_heatmap(data: DataFrame, plot_title: str = 'Correlation Matrix',
                             save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap with annotations.

    Args:
        data: Input DataFrame with numeric columns
        plot_title: Plot title
        save_path: Optional path to save figure
    """
    figure(figsize=(12, 8))

    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap with annotations
    heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    title(plot_title, fontsize=14, fontweight='bold')
    tight_layout()

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_pairplot(data: DataFrame, hue_column: Optional[str] = None,
                 save_path: Optional[str] = None) -> None:
    """
    Create pair plot for multivariate analysis.

    Args:
        data: Input DataFrame
        hue_column: Optional column name for color grouping
        save_path: Optional path to save figure
    """
    pair_plot = pairplot(data, hue=hue_column, diag_kind='kde',
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black'},
                        corner=True)

    pair_plot.fig.suptitle('Pair Plot of Energy Consumption Features',
                          y=1.02, fontsize=14, fontweight='bold')

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_residuals(true_values: ndarray, predicted_values: ndarray,
                   save_path: Optional[str] = None) -> None:
    """
    Plot residuals vs fitted values for regression diagnostics.

    Args:
        true_values: True values
        predicted_values: Predicted values
        save_path: Optional path to save figure
    """
    residuals = true_values - predicted_values

    figure(figsize=(10, 6))
    scatter(predicted_values, residuals, alpha=0.6, edgecolors='black', linewidths=0.5)
    axhline(y=0, color='r', linestyle='--', linewidth=2)

    title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
    xlabel('Fitted Values', fontsize=12)
    ylabel('Residuals', fontsize=12)
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_qq_plot(residuals: ndarray, save_path: Optional[str] = None) -> None:
    """
    Create Q-Q plot for normality check of residuals.

    Args:
        residuals: Residuals from regression
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))

    import matplotlib.pyplot as plt
    stats.probplot(residuals, dist="norm", plot=plt)

    title('Q-Q Plot', fontsize=14, fontweight='bold')
    xlabel('Theoretical Quantiles', fontsize=12)
    ylabel('Sample Quantiles', fontsize=12)
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_prediction_vs_actual(true_values: ndarray, predicted_values: ndarray,
                              save_path: Optional[str] = None) -> None:
    """
    Plot predicted vs actual values with diagonal line.

    Args:
        true_values: True values
        predicted_values: Predicted values
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))
    scatter(true_values, predicted_values, alpha=0.6, edgecolors='black', linewidths=0.5)

    # Add diagonal line (perfect prediction)
    min_value = min(true_values.min(), predicted_values.min())
    max_value = max(true_values.max(), predicted_values.max())
    plot([min_value, max_value], [min_value, max_value], 'r--', linewidth=2, label='Perfect Prediction')

    title('Predicted vs Actual Energy Consumption', fontsize=14, fontweight='bold')
    xlabel('Actual Values', fontsize=12)
    ylabel('Predicted Values', fontsize=12)
    legend()
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_elbow_curve(inertias: List[float], k_values: List[int],
                    save_path: Optional[str] = None) -> None:
    """
    Plot elbow curve for K-means clustering.

    Args:
        inertias: List of inertia values
        k_values: List of K values
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))
    plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)

    title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    xlabel('Number of Clusters (K)', fontsize=12)
    ylabel('Inertia', fontsize=12)
    grid(True, alpha=0.3)
    xticks(k_values)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()


def plot_clusters_2d(features_array: ndarray, cluster_labels: ndarray, feature1_index: int,
                     feature2_index: int, feature_names: List[str],
                     save_path: Optional[str] = None) -> None:
    """
    Plot 2D visualization of clusters.

    Args:
        features_array: Feature array
        cluster_labels: Cluster labels
        feature1_index: Index of first feature
        feature2_index: Index of second feature
        feature_names: List of feature names
        save_path: Optional path to save figure
    """
    figure(figsize=(10, 6))

    scatter_plot = scatter(features_array[:, feature1_index], features_array[:, feature2_index],
                          c=cluster_labels, cmap='viridis', alpha=0.6,
                          edgecolors='black', linewidths=0.5)

    colorbar(scatter_plot, label='Cluster')
    title('K-Means Clustering Results', fontsize=14, fontweight='bold')
    xlabel(feature_names[feature1_index], fontsize=12)
    ylabel(feature_names[feature2_index], fontsize=12)
    grid(True, alpha=0.3)

    if save_path:
        savefig(save_path, dpi=300, bbox_inches='tight')

    show()
