# California House Pricing Prediction (Part 2)

This repository documents the second and final phase of predicting California housing prices using **Scikit-Learn**, **Pandas**, **NumPy**, and **Joblib**. Building directly on the exploratory analysis and stratified sampling established in Part 1, Part 2 focuses on the complete machine learning engineering workflow: **categorical feature encoding**, **feature scaling**, **pipeline construction**, **ColumnTransformer consolidation**, **model training & cross-validation**, and **end-to-end model persistence and inference**.

---

## 📌 Project Overview & Workflow

The workflow progresses sequentially across 6 Jupyter Notebooks:

```
[11_HandlingCategoricalValues] ──▶ [12_FeatureScaling] ──▶ [13_ConstructingPipeline]
                                                                  │
                                                                  ▼
[16_ModelPersistanceAndInference] ◀── [15_TrainingMLAlgo] ◀── [14_ColumnTransformAndConsolidation]
```

### Key Progression:
1. **Handling Categorical Data:** Encoding `ocean_proximity` using `OrdinalEncoder` vs. `OneHotEncoder`.
2. **Feature Scaling:** Comparing `MinMaxScaler` (normalization) with `StandardScaler` (standardization).
3. **Pipeline Automation:** Packaging transformations into reusable Scikit-Learn `Pipeline` objects.
4. **Column Consolidation:** Unifying numerical and categorical branches into a single `ColumnTransformer`.
5. **Model Evaluation & Cross-Validation:** Training and comparing Linear Regression, Decision Tree, and Random Forest models with 10-fold cross-validation.
6. **Persistence & Inference:** Exporting the unified master pipeline via `joblib`, running batch predictions on unseen test data, and verifying final generalization performance.

---

## 📓 Notebook Walkthrough

### 1. `11_HandlingCategoricalValues.ipynb` — Categorical Encoding
- **Isolating Categorical Attributes:** Extracted `housing[["ocean_proximity"]]` as a 2D DataFrame (required by Scikit-Learn transformers).
- **Ordinal Encoding (`OrdinalEncoder`):** Explored integer mapping ($0, 1, 2, \dots$). Identified its fundamental limitation: machine learning algorithms assume nearby numbers are similar, which introduces unintended numerical relationships into unordered categories.
- **One-Hot Encoding (`OneHotEncoder`):** Implemented dummy binary feature encoding. Analyzed Scikit-Learn's default memory-efficient SciPy CSR sparse matrix representation and demonstrated conversion to dense arrays and DataFrames via `get_feature_names_out()`.

---

### 2. `12_FeatureScaling.ipynb` — Feature Scaling & Normalization
- **Why Scale:** Highlighted the vast scale disparity between attributes (e.g., `total_rooms` spanning up to 39,320 vs. `median_income` between 0.5 and 15.0).
- **Min-Max Scaling (`MinMaxScaler`):** Scaled features to a bounded range $[0, 1]$. Noted its sensitivity to extreme outliers.
- **Standardization (`StandardScaler`):** Rescaled features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), providing robust behavior against outliers.
- **Data Leakage Prevention:** Reinforced the golden rule: fit scalers **only** on the training set, never on test or combined datasets. Excluded target labels and binary dummy features from scaling.

---

### 3. `13_ConstructingPipeline.ipynb` — Sequential Preprocessing Pipelines
- **Pipeline Architecture:** Chained `SimpleImputer(strategy="median")` and `StandardScaler()` into a single `Pipeline` for numerical attributes.
- **Scikit-Learn Pipeline Rules:**
  - Specified name-estimator pairs `("name", transformer)`.
  - Enforced unique step names without double underscores `__`.
  - Ensured all intermediate steps implement `fit_transform()`.
- **Execution:** Demonstrated `pipeline.fit_transform(housing_num)` executing sequential imputation and scaling in a single automated step.

---

### 4. `14_ColumnTransformAndConsolidation.ipynb` — Full ColumnTransformer
- **Parallel Feature Processing:** Built a unified `ColumnTransformer` handling heterogeneous data types simultaneously:
  - Numerical pipeline applied to 8 numerical columns.
  - Categorical pipeline (`OneHotEncoder(handle_unknown="ignore")`) applied to `ocean_proximity`.
- **Dimensionality Verification:** Confirmed the transformed matrix landed at **13 features** (8 numerical + 5 one-hot columns) across 16,512 training instances $\rightarrow$ `(16512, 13)`.
- **Production Safety:** Leveraged `handle_unknown="ignore"` to safely handle unseen categorical levels during future inference.

---

### 5. `15_TrainingMLAlgo.ipynb` — Model Training & Cross-Validation
- **Models Evaluated:**
  1. `LinearRegression`: Baseline linear model.
  2. `DecisionTreeRegressor`: Nonlinear model capable of complex feature interactions.
  3. `RandomForestRegressor`: Ensemble of 100 decision trees.
- **Underfitting vs. Overfitting Diagnosis:**
  - The Decision Tree achieved a training RMSE of `0.0`, revealing complete memorization/overfitting.
  - 10-fold cross-validation (`cross_val_score` with `neg_root_mean_squared_error`) exposed that the Decision Tree actually performs no better than Linear Regression on unseen folds.
  - Random Forest dramatically outperformed both models.

#### Model Comparison Table (10-Fold CV):
| Model | 10-Fold CV RMSE (Mean) | CV Std Dev | Performance Diagnosis |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | **~$69,204** | ~$2,500 | Underfitting (High Bias) |
| **Decision Tree Regressor** | **~$69,348** | ~$2,110 | Severe Overfitting (High Variance) |
| **Random Forest Regressor** | **~$49,461** | ~$2,144 | **Best Generalization Performance** |

---

### 6. `16_ModelPersistanceAndInference.ipynb` — Master Pipeline & Inference
- **Master Pipeline Assembly:** Consolidated the entire preprocessing `ColumnTransformer` and the final `RandomForestRegressor` into one overarching end-to-end `Pipeline`.
- **Model Serialization (`joblib`):** Saved the complete pipeline (including imputer medians, scaler parameters, one-hot categories, and trained decision trees) into a single file: `housing_full_pipeline.pkl`.
- **Inference Pipeline:**
  - Loaded `housing_full_pipeline.pkl` from disk.
  - Read raw, un-preprocessed test instances from `input.csv`.
  - Called `loaded_pipeline.predict(X_test)` directly on raw inputs, running preprocessing and prediction in a single call.
- **Evaluation & Output:**
  - Calculated final test set RMSE against actual values.
  - Computed individual row-level prediction errors (`predicted - actual`).
  - Saved full predictions and error analysis to `output.csv`.

---

## 📂 Folder Structure

```
California house pricing prediction part 2/
│
├── 11_HandlingCategoricalValues.ipynb           # Ordinal vs One-Hot Encoding
├── 12_FeatureScaling.ipynb                     # MinMaxScaler vs StandardScaler & outlier impact
├── 13_ConstructingPipeline.ipynb                 # Sequential SimpleImputer + StandardScaler Pipeline
├── 14_ColumnTransformAndConsolidation.ipynb      # ColumnTransformer merging numerical + categorical
├── 15_TrainingMLAlgo.ipynb                       # Linear, Tree & Random Forest training & 10-fold CV
├── 16_ModelPersistanceAndInference.ipynb         # Master pipeline, joblib persistence, inference & test eval
├── housing.csv.xlsx                              # California housing dataset (source data)
├── housing_full_pipeline.pkl                     # Serialized end-to-end master pipeline (model + preprocessors)
├── input.csv                                     # Unseen test set used for simulating batch inference
├── output.csv                                    # Inference output with actuals, predictions, and errors
└── README.md                                     # Complete Part 2 documentation & learning notes
```

---

## 🛠️ Requirements & Environment

To run the notebooks and load the serialized pipeline locally:

```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

Launch Jupyter:

```bash
jupyter notebook
```

