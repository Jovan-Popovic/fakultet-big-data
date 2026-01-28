# Household Electrical Energy Consumption Analysis

A data science project analyzing household electrical energy consumption using statistical methods, machine learning, and data visualization.

## Project Overview

This project applies data science techniques to understand and predict household energy consumption patterns:

- **Descriptive Statistics**: Statistical profiling of energy consumption data
- **Data Visualization**: 15+ visualizations revealing patterns and relationships
- **Predictive Modeling**: OLS regression model using statsmodels
- **Pattern Discovery**: K-means clustering to identify consumption behavior groups
- **Business Insights**: Actionable recommendations for energy optimization

### Dataset

- **Training**: 1,000 samples
- **Test**: 100 samples
- **Features**: Square Footage, Number of Occupants, Appliances Used, Average Temperature, Building Type, Day of Week
- **Target**: Energy Consumption (kWh)

## Setup and Installation

### Prerequisites

- Python 3.13+
- Poetry 2.1+
- pyenv (for Python version management)

### Installation Steps

1. **Set Python version**:

   ```bash
   pyenv local 3.13.5
   ```

2. **Configure Poetry** to create virtual environment in project:

   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Activate environment**:

   ```bash
   poetry env activate
   ```

4. **Install dependencies**:

   ```bash
   poetry install
   ```

## Running the Analysis

The main analysis is in **`main.ipynb`** at the project root.

1. **Launch Jupyter**:

   ```bash
   poetry run jupyter notebook main.ipynb
   ```

2. **Run the analysis**:
   - Execute all cells: Cell → Run All
   - Or run cells sequentially to follow step-by-step

3. **View outputs**:
   - Visualizations saved to `outputs/` directory
   - Metrics and statistics saved as CSV/JSON files

## Project Structure

```
.
├── main.ipynb              # Main analysis notebook
├── datasets/               # Training and test CSV files
├── src/                    # Reusable Python modules
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── visualization.py   # Plotting functions
│   └── models.py          # ML models (OLS, K-means)
└── outputs/               # Generated plots and results
```

## Analysis Sections

1. **Data Loading & Exploration** - Load and inspect datasets
2. **Descriptive Statistics** - Mean, median, correlation, covariance, skewness, kurtosis
3. **Visualizations** - Histograms, scatter plots, box plots, heatmaps, pair plots
4. **Linear Regression** - OLS model with statsmodels, feature scaling, VIF analysis
5. **Clustering** - K-means with elbow method, cluster profiling
6. **Conclusions** - Business insights and recommendations

## Key Technologies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **statsmodels**: OLS regression
- **scikit-learn**: StandardScaler, K-means clustering
- **jupyter**: Interactive analysis environment

## Outputs

All generated files are saved to `outputs/`:

- 15+ visualization plots (PNG, 300 DPI)
- Model performance metrics (JSON)
- Cluster statistics (CSV)
- Test predictions with errors (CSV)
- VIF analysis results (CSV)

## Course Alignment

This project applies techniques from university data science curriculum:

- Descriptive statistics and correlation analysis
- OLS regression with statsmodels
- K-means clustering with elbow method
- Professional data visualization
