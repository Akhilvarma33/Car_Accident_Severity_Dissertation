# Car_Accident_Severity_Dissertation

Project Title:
# Car Accident Severity Prediction Using Random Forest, XGBoost, Logistic Regression, Decision Tree and Hierarchical Clustering with SMOTE and Time-Series Feature Integration

Dataset:
# Soundankar, A. (2022). Road Accidents Dataset. Kaggle.
# https://www.kaggle.com/datasets/atharvasoundankar/road-accidents-dataset

# Overview:
This project performs exploratory data analysis (EDA), feature engineering, preprocessing, model training, evaluation, and interpretation for predicting accident severity using the Road Accidents Dataset (Soundankar, 2022). The workflow is implemented in a Colab notebook (Car_Accident_Severity_Full_Code_Colab_Akhil.ipynb) and is designed to be reproducible both in Google Colab and local environments.

# Objectives:

Understand the structure and quality of the dataset through EDA and visualizations.

Engineer temporal and contextual features relevant to accident severity.

Handle missing values and class imbalance appropriately.

Train and compare several classification models (baseline and tree-based ensembles).

Evaluate models using classification metrics and ROC-AUC.

Interpret the best model using feature importance and SHAP values.

Provide clear instructions for reproducing the results.

Notebook / Files:

Car_Accident_Severity_Full_Code_Colab_Akhil.ipynb (main Colab notebook)

data/Road_Accidents_Data.csv 

requirements.txt (recommended Python packages)



Note: The notebook in Colab mounts Google Drive and reads the CSV from '/content/drive/MyDrive/Colab Notebooks/Road_Accidents_Data.csv' by default. If you run locally, place the CSV in the data/ directory or change the path accordingly.


Key Columns Observed (as used in the notebook):

Accident Date

DateTime

Time

Year

Month

Quarter

DayOfWeek

Hour

Is_Night

Is_Weekend

Number_of_Vehicles

Accident_Severity (target column; sometimes referenced as Severity)
Note: The dataset contains additional attributes (location, weather, road surface, casualty counts, vehicle types, etc.). Verify actual column names in your CSV before running.

Environment and Requirements:
Minimum recommended Python packages:

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

imbalanced-learn

shap

joblib

jupyterlab or notebook (if running locally)

Example pip install command:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap joblib jupyterlab

If you prefer a requirements.txt, include at least:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
joblib
jupyterlab

High-level Workflow (detailed explanation):

Data Loading

The notebook reads the CSV into a pandas DataFrame. In Colab, Google Drive is mounted and the CSV is loaded from the mounted path.

Immediately inspect the DataFrame shape, data types, and a few rows to confirm successful load.

Initial Cleaning

Standardize column names if necessary (trim whitespace, replace spaces with underscores or use consistent strings in code).

Convert date/time columns to pandas datetime objects using pd.to_datetime. This enables extraction of Year, Month, DayOfWeek, Hour, Quarter, and other time-related features.

Fix obvious typos in categorical values (for instance, any misspellings in severity labels).

Remove exact duplicates if they exist.

Missing Value Handling

Quantify missingness per column and decide imputation strategies:

Drop rows with missing target values.

For categorical fields with few missing values, impute with a new category like "Unknown" or the mode.

For numeric fields, impute using median or domain-appropriate values.

Record how many rows are affected by each imputation step to ensure transparency.

Feature Engineering

Temporal features: extract Year, Month, DayOfWeek, Hour, Quarter from datetime fields.

Derived flags: create Is_Night (based on Hour), Is_Weekend (based on DayOfWeek), and potentially rush-hour flags.

Aggregations: if location identifiers exist, compute aggregated statistics (e.g., average severity by location) if useful and privacy-safe.

Binning: if any continuous variable has skew or outliers (for example driver age or casualty counts), consider binning into categories or winsorizing.

Exploratory Data Analysis (EDA)

Univariate analysis: distribution plots for numeric variables and bar charts for categorical variables.

Bivariate analysis: cross-tabulations and boxplots to inspect relationships between Accident_Severity and predictors.

Correlation analysis: numeric correlation matrix and heatmap to find multicollinearity.

Visualize class distribution of the target to understand imbalance.

Handling Class Imbalance

The notebook uses SMOTE (Synthetic Minority Over-sampling Technique) from imbalanced-learn to oversample minority classes in the training set.

Alternative strategies: class weighting in model training, random undersampling, or stratified sampling. SMOTE is effective when synthetic samples are reasonable for the feature space.

Encoding and Scaling

Categorical encoding:

For nominal categorical variables, use one-hot encoding (get_dummies) or sklearn OneHotEncoder.

For ordinal features, use LabelEncoder or map to an ordered integer scale if order is meaningful.

Scaling:

StandardScaler is applied to numeric features when required (e.g., for models sensitive to feature scales like Logistic Regression or KNN).

Tree-based models (RandomForest, XGBoost) do not require scaling but standardization does not harm and keeps preprocessing consistent.

Train / Test Split

Use train_test_split with a reproducible random_state and stratify on the target to preserve class proportions.

Typical split: 70–80% training, 20–30% testing. Optionally use a validation split or cross-validation for hyperparameter tuning.

Models Trained

Baseline: Logistic Regression (for a simple interpretable baseline).

Tree-based models: Decision Tree, Random Forest, XGBoost.

For each model:

Fit on training data.

Predict on test data and compute probabilities for ROC/AUC.

Optionally tune hyperparameters using GridSearchCV or RandomizedSearchCV (not mandatory in the notebook).

Evaluation Metrics and Interpretation

Classification metrics:

Accuracy: overall correct predictions (useful but can be misleading on imbalanced data).

Precision and Recall: per-class precision and recall to understand false positives and false negatives.

F1-score: harmonic mean of precision and recall; useful when class balance is uneven.

Confusion Matrix: visual matrix to inspect per-class prediction errors.

ROC-AUC: compute multiclass ROC curves and AUC (one-vs-rest) when appropriate.

Model Comparison:

Compare models using the same metrics and prefer models that balance precision and recall for the classes of interest.




How to Run (Colab):

Open Car_Accident_Severity_Full_Code_Colab_Akhil.ipynb in Google Colab.

Mount Google Drive when prompted:
from google.colab import drive
drive.mount('/content/drive')

Ensure the dataset CSV is available at:
/content/drive/MyDrive/Colab Notebooks/Road_Accidents_Data.csv
or modify the read_csv path in the notebook to match your location.

Run cells sequentially. Large models or SHAP calculations may take additional time in Colab runtime.

How to Run (Local):

Create a Python environment and install requirements (see requirements.txt).

Place Road_Accidents_Data.csv in data/ or update the notebook/script paths.

Launch JupyterLab or Jupyter Notebook:
jupyter lab

Open the notebook and run cells. Alternatively, convert notebook cells into a script and run via command line.

Reproducibility Notes:

Use random_state in train_test_split and model initializers to reproduce results.

Use saved versions of preprocessed datasets and trained models for exact replication.

Record package versions (pip freeze > requirements-freeze.txt) if strict reproducibility is required.

Expected Outputs:

Descriptive EDA plots (histograms, boxplots, heatmaps).

Table of model performance metrics (accuracy, precision, recall, f1, ROC-AUC).

Confusion matrices for each model.

SHAP summary plot and dependence plots for the final model.

Saved model artifact(s) and example predictions on holdout/test set.

Limitations and Caveats:

Quality of model predictions depends on the completeness and correctness of recorded features in the dataset.

Geographic or temporal biases in the dataset may limit generalizability.

Synthetic oversampling (SMOTE) can introduce unrealistic samples if features are not well-behaved; inspect synthetic examples when used.

Privacy: be mindful of any personally identifiable information (PII). Do not expose raw location identifiers or PII in public outputs.

Extensions and Future Work:

Incorporate additional context features such as traffic volume, weather APIs, or road network data.

Use cross-validation and more extensive hyperparameter tuning.

Explore more advanced models (LightGBM, CatBoost) and ensemble stacking.

Build a small web app or API for model inference and visualization.

Citation:

Soundankar, A. (2022). Road Accidents Dataset. Kaggle. https://www.kaggle.com/datasets/atharvasoundankar/road-accidents-dataset



