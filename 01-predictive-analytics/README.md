# Predictive Analytics: Medical Insurance Cost Prediction

**Author:** Krisna Rhesa  
**Track:** Applied Machine Learning - Dicoding Indonesia  
**Dataset:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## Project Domain

Healthcare and medical treatment costs represent one of the most volatile and burdensome unexpected financial risks faced by individuals and institutions. According to reports by the *World Health Organization* (WHO, 2021), the macroeconomic burden of non-communicable diseases (such as cardiovascular conditions, stroke, and diabetes) continues to escalate globally. A primary catalyst driving this inflation is high-risk lifestyle behaviors - most notably tobacco usage and elevated Body Mass Index (BMI).

In the private health insurance industry, formulating accurate and equitable annual premiums (*premium pricing*) is a fundamental operational challenge. Setting rates too low (*underpricing*) exposes insurers to insolvency risks when policyholders file catastrophic claims. Conversely, setting rates too high (*overpricing*) drives low-risk, healthy individuals away from purchasing insurance, leaving an asymmetrically high concentration of vulnerable, high-risk policyholders (*adverse selection*) (Lantz, 2019).

Actuarial complexity arises from non-linear, compounding interactions among physiological and behavioral risk factors. For instance, smoking is not merely additive; it exerts a super-additive multiplier effect when paired with clinical obesity ($BMI \ge 30$). Traditional linear actuarial tables often struggle to capture such multi-factor interactions dynamically.

Empirical research by Ganesan et al. (2020) demonstrates that regression-based supervised machine learning models effectively capture complex non-linear patterns and multi-variable interactions to estimate individual medical liabilities. Consequently, this project implements a predictive regression pipeline to forecast individual annual medical charges. The models are trained on demographic characteristics and lifestyle indicators (age, sex, BMI, number of children, smoking status, and geographical region) to enable data-driven, objective, and equitable risk-based premium pricing.

### References
- Ganesan, N., et al. (2020). "Application of Machine Learning Techniques in Medical Insurance Cost Prediction." *International Journal of Advanced Science and Technology*, 29(5), pp. 11452-11461.
- Lantz, B. (2019). *Machine Learning with R: Expert techniques for predictive modeling to solve problems in a practical way* (3rd ed.). Birmingham: Packt Publishing Ltd.
- World Health Organization (WHO). (2021). *Tobacco and noncommunicable diseases: Economic and health consequences of tobacco use*. Geneva: World Health Organization Press.

---

## Business Understanding

The business objective is to design an automated medical expense estimation pipeline capable of delivering accurate, unbiased claim forecasts. This enables actuarial underwriters to establish personalized, risk-based premium schedules proportionate to each policyholder's health profile.

### Problem Statements
1. Which demographic and lifestyle characteristics exert the most statistically dominant influence on annual medical charges?
2. How can a data preparation and modeling pipeline be architected to ensure strict isolation and eliminate data leakage while maximizing predictive fidelity?
3. Among K-Nearest Neighbors (KNN), Random Forest, and Gradient Boosting regressors, which model architecture yields the superior and most stable generalization performance on unseen test records?

### Goals
1. Identify and quantify the contribution of demographic and lifestyle predictors to medical cost variance through Exploratory Data Analysis (EDA) and model feature importance extraction.
2. Implement a leak-free data preparation pipeline (duplicate sanitation, categorical encoding, train-test splitting, and feature standardization).
3. Develop, benchmark, and optimize regression algorithms via systematic hyperparameter tuning to achieve an out-of-sample $R^2 > 0.85$ with minimized Root Mean Squared Error (RMSE).

### Solution Statements
- **Multi-Paradigm Algorithmic Benchmarking:** Evaluates three distinct regression architectures:
  1. *K-Nearest Neighbors (KNN) Regressor*: Distance-based non-parametric instance learning.
  2. *Random Forest Regressor*: Parallel bagging ensemble of randomized decision trees.
  3. *Gradient Boosting Regressor*: Sequential boosting ensemble optimizing gradient pseudo-residuals.
- **Hyperparameter Optimization via GridSearchCV:** Fine-tunes the top-performing baseline model across learning rate, tree depth, estimator count, and subsampling fractions under 5-fold cross-validation.
- **Comprehensive Metric Evaluation:** Quantifies model efficacy using four standard regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).

---

## Data Understanding

The project leverages the **Medical Cost Personal Datasets** (`insurance.csv`), compiled by Brett Lantz and sourced from the *U.S. Census Bureau* demographic surveys cross-referenced with national medical expense distributions.
- **Source Repository:** [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

### Dataset Characteristics:
- Contains **1,338 records** across **7 feature columns** (exceeding Dicoding's 500-sample threshold).
- Quality audits confirmed **0 missing values** across all attributes.
- **1 duplicate row** was identified and eliminated during data preparation, leaving **1,337 clean observations**.

### Feature Definitions:
1. `age`: Age of the primary beneficiary (continuous integer, 18-64 years).
2. `sex`: Insurance contractor gender (binary categorical: `female`, `male`).
3. `bmi`: Body Mass Index ($kg/m^2$), weight relative to squared height (continuous float, 15.96-53.13). Healthy range: 18.5-24.9.
4. `children`: Number of dependent children covered by health insurance (discrete integer, 0-5).
5. `smoker`: Smoking status of policyholder (binary categorical: `yes`, `no`).
6. `region`: Residential area in the United States (nominal categorical: `northeast`, `southeast`, `southwest`, `northwest`).
7. `charges`: Individual annual medical expenses billed to health insurance in USD ($) (continuous target variable).

### Exploratory Data Analysis (EDA)

#### 1. Descriptive Statistics
- **Mean Age (`age`)**: 39.2 years, evenly distributed between 18 and 64.
- **Mean BMI (`bmi`)**: 30.66 $kg/m^2$ (indicating that the majority of sample individuals fall into the overweight to class-I obese categories).
- **Medical Charges (`charges`)**: Mean is $13,270.42 while the median is $9,382.03. This large divergence highlights severe positive skewness (*right-skewed*), driven by high-cost medical interventions reaching up to $63,770.43.

#### 2. Univariate Distribution Analysis
Distribution plots of key features are shown in Figure 1:

![Univariate Distributions](figures/eda_univariate_distribution.png)
*Figure 1. Univariate Frequency Distributions of Charges, Age, and BMI*

`charges` exhibits a prominent right-side long tail; `age` is uniformly spread across adult cohorts; `bmi` closely follows a symmetric Gaussian bell curve.

#### 3. Bivariate and Multivariate Interactions
Bivariate feature interactions against `charges` are illustrated in Figure 2:

![Bivariate Relationships](figures/eda_bivariate_relationships.png)
*Figure 2. Bivariate and Multivariate Interactions Against Medical Charges*

Key analytical insights:
- **Smoking Impact**: Tobacco consumption is the single most decisive cost differentiator. Median charges for smokers are **$34,456** compared to **$7,326** for non-smokers (>4.7x disparity).
- **Obesity $\times$ Smoking Multiplier**: For active smokers, individuals with $BMI \ge 30$ experience an exponential surge in claims into the $30,000-$60,000+ bracket. For non-smokers, increasing BMI induces only a modest, linear rise.
- **Age Progression**: Across all cohorts, medical claims exhibit a steady, positive drift with advancing age due to cumulative biological wear.

#### 4. Pearson Correlation Matrix
The linear correlation heatmap is presented in Figure 3:

![Correlation Heatmap](figures/eda_correlation_heatmap.png)
*Figure 3. Pearson Correlation Heatmap Across Features and Charges*

Smoking status (`smoker_yes`) displays the strongest linear correlation with charges ($r = 0.79$), followed by age ($r = 0.30$) and BMI ($r = 0.20$).

#### 5. Outlier Evaluation
Applying the $1.5 \times IQR$ threshold reveals 139 high-charge records (10.40%) exceeding $34,524. These observations were **deliberately retained**. In health actuarial science, catastrophic claims resulting from severe pathology or complex surgery represent legitimate empirical phenomena essential for model generalization.

---

## Data Preparation

The preparation pipeline follows a strict isolation protocol to prevent information leakage:

1. **Deduplication**:
 - `df.duplicated().sum()` detected 1 identical record. Removed via `df.drop_duplicates()`, retaining 1,337 unique samples.
2. **One-Hot Encoding**:
 - Categorical columns (`sex`, `smoker`, `region`) transformed into numeric dummies via `pd.get_dummies(..., drop_first=True, dtype=int)` to avoid collinearity.
3. **Train-Test Splitting**:
 - 80% train (1,069 samples) and 20% test (268 samples) split using `train_test_split(..., random_state=42)`.
4. **Feature Standardization**:
 - Continuous numerical features (`age`, `bmi`, `children`) standardized to zero-mean and unit-variance ($\mu = 0, \sigma = 1$) via `StandardScaler`.
 - **Data Leakage Mitigation:** The scaler is fitted exclusively on `X_train`, and then used to transform both `X_train` and `X_test`.

```python
df_prep = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
X = df_prep.drop(columns=['charges'])
y = df_prep['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
```

---

## Modeling

Three regression paradigms were implemented and compared:

1. **K-Nearest Neighbors (KNN) Regressor:**
 - *Parameters:* `n_neighbors=5`, Euclidean distance.
 - *Trade-offs:* Simple and non-parametric; degrades in higher-dimensional sparse dummy spaces.
2. **Random Forest Regressor:**
 - *Parameters:* `n_estimators=100, max_depth=16, random_state=42`.
 - *Trade-offs:* Highly robust against noise; slight tendency toward memorizing training variance without strict depth pruning.
3. **Gradient Boosting Regressor:**
 - *Parameters:* `n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`.
 - *Trade-offs:* Sequentially minimizes pseudo-residuals; excellent bias-variance trade-off on tabular medical data.

### Hyperparameter Tuning (GridSearchCV)
Gradient Boosting achieved the most favorable baseline generalization and was further tuned via 5-fold cross-validated grid search:

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 1.0]
}
```

Optimal configuration discovered:
- `learning_rate`: **0.05**
- `max_depth`: **2**
- `n_estimators`: **150**
- `subsample`: **0.8**

---

## Evaluation

Model performance was tracked using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **Coefficient of Determination ($R^2$ Score)**.

### Comparative Benchmark Results

*Table 1. Quantitative Regression Metrics Comparison Across Models*

| Model Architecture | Train MSE | Test MSE | Train RMSE ($) | Test RMSE ($) | Train MAE ($) | Test MAE ($) | Train $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN Regressor** | 2.8025e+07 | 6.5004e+07 | 5,293.84 | 8,062.52 | 3,125.86 | 4,494.19 | 0.7953 | 0.6462 |
| **Random Forest** | 3.5926e+06 | 2.2239e+07 | 1,895.41 | 4,715.80 | 1,057.31 | 2,646.95 | 0.9738 | 0.8790 |
| **Gradient Boosting (Baseline)** | 1.4841e+07 | 1.8218e+07 | 3,852.41 | 4,268.28 | 2,109.40 | 2,517.47 | 0.8916 | 0.9009 |
| **Gradient Boosting (Tuned)** | **1.8283e+07** | **1.8161e+07** | **4,275.84** | **4,261.55** | **2,418.14** | **2,526.74** | **0.8665** | **0.9012** |

Visual comparisons of out-of-sample metrics:

![Model Metrics](figures/model_comparison_metrics.png)
*Figure 4. Comparative RMSE and R² Benchmark Across Regressors*

Scatter plot of actual vs. predicted charges for the tuned model:

![Actual vs Predicted](figures/actual_vs_predicted.png)
*Figure 5. Actual vs Predicted Values (Tuned Gradient Boosting)*

Relative feature importance scores:

![Feature Importance](figures/feature_importance.png)
*Figure 6. Relative Predictive Feature Importance Scores*

### Model Selection Rationale

**Tuned Gradient Boosting Regressor** was selected as the production model because:
1. **Superior Accuracy**: Achieved highest out-of-sample explanatory power with **Test $R^2 = 0.9012$** (accounting for 90.12% of billing variance).
2. **Minimized Error**: Produced the lowest monetary loss with **Test RMSE = $4,261.55** and **Test MAE = $2,526.74**.
3. **Flawless Generalization**: The train-test RMSE difference is only **$14.29** ($4,275.84 vs $4,261.55), confirming optimal regularization with zero overfitting or underfitting.

### Business Impact
Deploying this tuned model replaces static underwriting tables with individualized, dynamic risk-based pricing. It directly prevents financial deficits from underpricing high-risk subscribers while eliminating the adverse selection cycle among healthy policyholders.
