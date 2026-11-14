# Engage 2 Value: From Clicks to Conversions

Machine Learning Practice (MLP) project notebook for the Kaggle competition
**"Engage 2 Value: From Clicks to Conversions"**. The task is to predict the
`purchaseValue` of a user session on an e-commerce site from web-analytics
data: traffic source, device, geography and engagement metrics.

Notebook: [`23f1002689-notebook-t22025.ipynb`](23f1002689-notebook-t22025.ipynb)

## Data

| Split | Rows    | Columns |
|-------|---------|---------|
| Train | 116,023 | 52      |
| Test  | 29,006  | 51      |

The target is mostly zeros. The 25th, 50th and 75th percentiles are all 0,
because most sessions don't buy anything. Among the sessions that do buy, the
values are extremely right-skewed (skew ≈ 53.9).

## What the notebook does

### 1. Exploratory data analysis
- Splits the features into categorical, numerical and boolean columns (37 / 14 / 1).
- Finds columns with more than 60% missing values, and columns with a single constant value.
- Checks which numerical features correlate with `purchaseValue`.
- Looks at `totalHits` and `pageViews` one at a time and against the target,
  and compares skew-reducing transforms (log, sqrt, cube root, reciprocal,
  Box-Cox, Yeo-Johnson). The reciprocal and Box-Cox transforms reduce skew the most.

### 2. Feature engineering
- **Date-level features:** `PurchasesPerDate` (purchases made on each date) and
  `PCR` (purchase conversion rate per date).
- **Engagement features:** page views per hit, and log, reciprocal and Box-Cox
  transforms of hits and page views.
- **Datetime features:** year, month, day, weekday and an `is_weekend` flag.
- **User aggregates:** per-user sums and means of hits and page views, plus hits per session number.
- **User level:** each session is labelled New / Returning / Loyal / Max Loyal by its session number.
- **Target encoding:** each user's mean `purchaseValue` (`user_purchase_mean`).

### 3. Preprocessing
- Drops columns that are mostly missing and columns with a constant value.
- Fills `totals.bounces` and `new_visits` with 0 where missing, since a missing value there means 0.
- Groups the rare categories of high-cardinality columns (browser, campaign, region,
  traffic source, OS, sub-continent, country, city, metro) into an "other" bucket.
- Uses a `ColumnTransformer` with `RobustScaler` for numerical columns and `OneHotEncoder` for categorical ones.

### 4. Modelling
The models are compared on sessions with a purchase (`purchaseValue > 0`), using an 80/20 train/validation split.

**Baseline comparison (validation R²):**

| Model                     | R²     |
|---------------------------|--------|
| Gradient Boosting         | 0.855  |
| XGBoost                   | 0.842  |
| Random Forest             | 0.834  |
| LightGBM                  | 0.731  |
| Linear Regression         | 0.480  |
| Lasso                     | 0.480  |
| Ridge                     | 0.479  |
| AdaBoost                  | -2.924 |

**After hyperparameter tuning** (`RandomizedSearchCV`, 25 iterations, 2-fold CV):

| Model             | R²    |
|-------------------|-------|
| XGBoost           | 0.876 |
| Gradient Boosting | 0.836 |
| Random Forest     | 0.709 |

### 5. Key insights
- Tree-based models do far better than linear ones. This suggests the relationships
  between the features and purchase value are non-linear.
- AdaBoost does badly because it gives more weight to the samples it gets wrong.
  That makes it very sensitive to the large outliers in `purchaseValue`.
- XGBoost was the best model after tuning, and it also trained fastest on a dataset this size.

### 6. Submissions
The notebook refits the tuned XGBoost, Gradient Boosting and Random Forest pipelines
on the full training set. It clips negative predictions to 0 and writes the submission
files `xgb.csv`, `gbr.csv` and `rf.csv`.

## Notes and limitations
- The R² scores above were measured only on sessions with a purchase. The final
  models are trained on all sessions, including those with zero purchase value,
  so the scores don't describe how the models do on the full test distribution.
- `user_purchase_mean` and the date-level purchase features are calculated from the
  whole training set before the validation split. This leaks target information into
  validation, so the validation scores are probably too optimistic.

## Running it
The notebook was written for Kaggle and reads its data from
`/kaggle/input/engage-2-value-from-clicks-to-conversions/`. To run it elsewhere,
download the competition data and change the paths in the second cell.

Requirements: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`.
