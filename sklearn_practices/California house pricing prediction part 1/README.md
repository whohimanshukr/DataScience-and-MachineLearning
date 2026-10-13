# California House Pricing Prediction (Part 1)

This project documents the first phase of predicting California housing prices using **Scikit-Learn**, **Pandas**, **NumPy**, and **Matplotlib**. It covers everything from initial data exploration (EDA) to stratified sampling, spatial visualization, feature correlation analysis, and early-stage data cleaning using Scikit-Learn's `SimpleImputer`.

---

## 📌 Project Overview & Workflow

The workflow is broken down sequentially across 5 Jupyter Notebooks:

```
[6_Analyzing_Housing_data] ──▶ [7_TestSet_RandomPer_Housing_data] ──▶ [8_TestSet_StrataSplit_Housing_data]
                                                                                   │
                                                                                   ▼
[10_FurtherPreprocessing_Housing_data] ◀── [9_Visualize_StrataSplit_Housing_data-Copy1]
```

### 1. Dataset & Acquisition
- **Source:** California Housing dataset (originally from the 1990 California census, available on Kaggle).
- **Data file:** `housing.csv.xlsx` (20,640 rows × 10 columns).
- **Features:**
  - `longitude` & `latitude`: Geographical coordinates.
  - `housing_median_age`: Median age of houses in a block.
  - `total_rooms`: Total number of rooms in a block.
  - `total_bedrooms`: Total number of bedrooms (contains missing values).
  - `population`: Total number of residents living in a block.
  - `households`: Total number of households (groups of people residing in a home).
  - `median_income`: Median income for households in tens of thousands of USD ($10,000s).
  - `median_house_value`: Target variable — median house value in USD (capped at $500,001).
  - `ocean_proximity`: Categorical feature describing location relative to the ocean (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`).

---

## 📓 Notebook Walkthrough

### 1. `6_Analyzing_Housing_data.ipynb` — Exploratory Data Analysis (EDA)
- **Data Ingestion:** Loaded `housing.csv.xlsx` using Pandas (`pd.read_excel`).
- **Initial Inspection:** Checked the first and last rows using `.head()` and `.tail()`.
- **Structural Analysis:** Used `.info()` to identify data types and inspect missing values (observed 207 missing values in `total_bedrooms`).
- **Categorical Breakdown:** Used `.value_counts()` on `ocean_proximity` to see district distribution across categories.
- **Statistical Summary:** Analyzed summary statistics via `.describe()` (count, mean, standard deviation, min, percentiles, max).
- **Data Distributions:** Generated 50-bin histograms (`df.hist(bins=50, figsize=(12,8))`) to visualize distributions, identifying right-skewed attributes and value caps on `housing_median_age` and `median_house_value`.

---

### 2. `7_TestSet_RandomPer_Housing_data.ipynb` — Random Permutation Splitting
- **Preventing Data Snooping Bias:** Emphasized creating a test set early before discovering unintended patterns.
- **Custom Shuffling Implementation:** Implemented a train-test split function from scratch using `np.random.permutation`:
  ```python
  def shuffle_and_split(data, test_ratio):
      np.random.seed(42)
      shuffled_indices = np.random.permutation(len(data))
      test_set_size = int(len(data) * test_ratio)
      test_indices = shuffled_indices[:test_set_size]
      train_indices = shuffled_indices[test_set_size:]
      return data.iloc[train_indices], data.iloc[test_indices]
  ```
- **Result:** Split the dataset into 80% training (16,512 instances) and 20% test (4,128 instances).
- **Limitation Identified:** Purely random sampling can introduce sampling bias if key demographic predictors (like median income) are not proportionately represented in both subsets.

---

### 3. `8_TestSet_StrataSplit_Housing_data.ipynb` — Stratified Sampling (Strata Split)
- **Income Categorization:** Because `median_income` is a continuous attribute strongly linked to house prices, discretized it into 5 distinct income strata using `pd.cut()`:
  - Bins: `[0, 1.5, 3.0, 4.5, 6.0, np.inf]`
  - Labels: `[1, 2, 3, 4, 5]`
- **Stratified Splitting:** Leveraged Scikit-Learn's `StratifiedShuffleSplit` to ensure both train and test sets maintain identical category proportions to the full dataset:
  ```python
  from sklearn.model_selection import StratifiedShuffleSplit

  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  for train_index, test_index in split.split(df, df["income_cat"]):
      strat_train_set = df.loc[train_index]
      strat_test_set = df.loc[test_index]
  ```
- **Verification:** Compared income category distribution proportions across the full dataset, random test set, and stratified test set to demonstrate reduced sampling error.

---

### 4. `9_Visualize_StrataSplit_Housing_data-Copy1.ipynb` — Spatial Visualization & Attribute Combinations
- **Data Preparation:** Removed the temporary `income_cat` column and worked strictly on a copy of the stratified training set (`strat_train_set.copy()`).
- **Geographical Plotting:**
  - Plotted latitude and longitude scatter plots.
  - Used `alpha=0.1` to visualize high-density population clusters (Bay Area, Los Angeles, San Diego, Central Valley).
  - Enhanced the scatter plot with bubble sizing representing population (`s=df["population"]/100`) and color mapping representing house prices (`c="median_house_value"`, `cmap="jet"`).
- **Correlation Analysis:** Computed Pearson's correlation coefficient (`corr_matrix["median_house_value"].sort_values(ascending=False)`), confirming `median_income` as the highest positive correlation (~0.687).
- **Feature Engineering:** Created experimental attribute combinations:
  - `rooms_per_household = total_rooms / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms` (demonstrated negative correlation: fewer bedrooms relative to total rooms correlates with higher home values)
  - `population_per_household = population / households`

---

### 5. `10_FurtherPreprocessing_Housing_data.ipynb` — Early Preprocessing & Imputation
- **Separating Features and Target:**
  - Predictors: `housing = strat_train_set.drop("median_house_value", axis=1)`
  - Labels: `housing_labels = strat_train_set["median_house_value"].copy()`
- **Handling Missing Values:** Handled nulls in `total_bedrooms` using Scikit-Learn's `SimpleImputer`:
  ```python
  from sklearn.impute import SimpleImputer
  import numpy as np

  imputer = SimpleImputer(strategy="most_frequent") # or median
  housing_num = housing.select_dtypes(include=[np.number])
  imputer.fit(housing_num)
  X = imputer.transform(housing_num)
  ```
- **Model Parameters Inspection:** Checked `imputer.statistics_` across all numerical columns and transformed the dataset into a complete, clean NumPy array ready for downstream pipeline transformation.

---

## 📂 Folder Structure

```
California house pricing prediction part 1/
│
├── 6_Analyzing_Housing_data.ipynb                  # EDA, data shapes, info, describe & histograms
├── 7_TestSet_RandomPer_Housing_data.ipynb          # Random permutation train-test split
├── 8_TestSet_StrataSplit_Housing_data.ipynb        # Stratified sampling based on income categories
├── 9_Visualize_StrataSplit_Housing_data-Copy1.ipynb# Geographic map plotting, correlations & feature engineering
├── 10_FurtherPreprocessing_Housing_data.ipynb      # Separating labels & missing value imputation via SimpleImputer
├── housing.csv.xlsx                                # California housing dataset (Excel format)
└── README.md                                       # Documentation & learning notes
```

---

## 🛠️ Requirements & Environment

To run the notebooks locally, install the necessary libraries:

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
```

Launch Jupyter:

```bash
jupyter notebook
```
