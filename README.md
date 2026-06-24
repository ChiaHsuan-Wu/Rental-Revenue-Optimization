# Rental Revenue Prediction Pipeline
**Predictive Analytics & Machine Learning | Rental Revenue Forecasting Optimization**

### Project Overview
This repository features an end-to-end machine learning pipeline developed for a competitive rental revenue prediction challenge on **Codalab**. The pipeline implements a series of regression models to forecast listing revenue based on spatial (lat/lon), property (rooms, beds, bathrooms), and facility-related features. The solution emphasizes robust preprocessing, feature engineering, and iterative model optimization.

### Tech Stack & Methodology
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, SciPy (for distribution sampling)
- **Pipeline Architecture:** `ColumnTransformer` & `Pipeline` integration for modular preprocessing (imputation, one-hot encoding).
- **Optimization Strategy:** 
    - **Baseline:** Ridge Regression & DummyRegressor.
    - **Advanced Ensembles:** RandomForest, GradientBoosting, and XGBoost.
    - **Tuning:** RandomizedSearchCV with 5-fold Cross-Validation for hyperparameter optimization.

### Pipeline Workflow
The project follows a reproducible data science lifecycle:
- `baseline.py`: Implements initial data ingestion, basic imputation, and Ridge Regression baseline performance.
- `optimized_model.py`: Performs comprehensive preprocessing (OneHotEncoding for categorical features, median/most_frequent imputation), robust model training, and hyperparameter tuning using `RandomizedSearchCV` to minimize Mean Absolute Error (MAE).

### Key Technical Highlights
- **Error Minimization:** Utilized logarithmic transformation (`np.log1p`) on the target variable to normalize revenue distributions, ensuring better performance for non-linear regression models.
- **Robust Cross-Validation:** Implemented K-Fold cross-validation to assess the stability of revenue predictions and prevent overfitting.
- **Hyperparameter Tuning:** Conducted extensive random searches across depth, estimators, and learning rates for gradient-boosted models, significantly outperforming baseline regression architectures.
