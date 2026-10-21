# Bayesian Logistic Regression for Heart Disease Prediction

> Bachelor's thesis · **Mikhail Fedorov** · **[University of Economics and Business / Faculty of Informatics and Statistics / Department of Data Analysis]** · **2025**
> Supervisor: **Ph.D. Tomáš Karel**

## Overview

This project compares **classical (frequentist)** and **Bayesian** logistic regression on a binary
classification task: predicting whether a patient has heart disease (`HeartDisease`).

The central question of the thesis is how the behavior of a Bayesian model depends on the choice of
prior distributions and the amount of available data. To study this, the original dataset is split into two parts:

| Part | Share | Size | Purpose |
|------|-------|------|---------|
| "Larger" dataset (`larger_dataset`) | 20 % | 183 × 16 | Feature selection, model training and evaluation, posterior estimates |
| "Smaller" dataset (`smaller_dataset`) | 80 % | 735 × 16 | Reserved for the next stage (further fitting / comparison with informative priors) |

> ⚠️ **Naming note.** In the code, `larger_dataset` is the first 20 % of the shuffled data (183 rows),
> while `smaller_dataset` is the remaining 80 % (735 rows). The names are inverted relative to the actual sizes.
> If the thesis text uses different terminology, consider aligning the README and the notebook.

## Data

`processed_heart_data.csv` is a preprocessed version of the **Heart Failure Prediction** dataset
(918 observations, 11 original features). After one-hot encoding there are 15 features plus the target.

**[Add the data source, e.g. a Kaggle/UCI link, and its license.]**

| Group | Features |
|-------|----------|
| Numeric (standardized) | `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak` |
| Binary | `Sex`, `FastingBS`, `ExerciseAngina` |
| Categorical (one-hot) | `ChestPainType_*`, `RestingECG_*`, `ST_Slope_*` |
| Target | `HeartDisease` (0 = healthy, 1 = heart disease) |

## Notebook Structure: `Large_dataset.ipynb`

1. **Installation and data preprocessing**
   - data loading, fixed `RANDOM_SEED = 42`;
   - splitting into two parts (20 % / 80 %);
   - train/test split of the larger part (75 / 25, stratified by `HeartDisease`) and standardization of numeric features (`StandardScaler`, fitted on the training set only).
2. **Analysis on the "larger" dataset**
   - classical logistic regression on all features (`statsmodels.Logit`): coefficients, standard errors, z-statistics, p-values, confidence intervals;
   - selection of significant features;
   - 5-fold cross-validation (`KFold`, `sklearn.LogisticRegression`);
   - final classical model on significant features only, with metrics and visualizations (ROC curve, confusion matrix).
3. **Bayesian logistic regression (Pyro)**
   - conversion of data to PyTorch tensors;
   - model with **non-informative priors**:
     `weights ~ Normal(0, 10)`, `bias ~ Normal(0, 10)`, `y ~ Bernoulli(logits)`;
   - inference via **MCMC (NUTS)**: 4 chains, 200 warm-up steps, 1000 samples;
   - convergence diagnostics (`n_eff`, `r_hat`), posterior means and 95 % credible intervals;
   - predictions by averaging over posterior samples, metrics, ROC curve, confusion matrix, trace plots, posterior density plots;
   - final comparison table.

## Selected Features

`Sex`, `Cholesterol`, `FastingBS`, `ExerciseAngina`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ST_Slope_Flat`

## Results

Test set: 46 observations.

| Metric | Classical LR | Bayesian LR (non-informative prior) | Bayesian LR (informative prior) |
|--------|:------------:|:-----------------------------------:|:-------------------------------:|
| Accuracy  | 0.78 | 0.78 | 0.83 |
| Precision | 0.81 | 0.81 | 0.76 |
| Recall    | 0.81 | 0.81 | 0.96 |
| F1-score  | 0.81 | 0.81 | 0.85 |
| ROC-AUC   | 0.81 | 0.81 | 0.93 |

> The results for informative priors are included in the notebook as a ready-made table
> (they are not computed in this file). **[State which notebook/script produced them and how the priors were defined.]**

Posterior coefficient estimates (non-informative priors, standardized features):

| Parameter | Mean | 95 % CI |
|-----------|:----:|:-------:|
| `bias` | −2.89 | [−5.08, −1.01] |
| `Sex` | 2.71 | [0.95, 4.73] |
| `Cholesterol` | −0.52 | [−1.26, 0.26] |
| `FastingBS` | 1.76 | [0.06, 3.44] |
| `ExerciseAngina` | 2.55 | [1.16, 4.15] |
| `ChestPainType_ATA` | −4.45 | [−6.88, −2.33] |
| `ChestPainType_NAP` | −2.10 | [−3.55, −0.70] |
| `ST_Slope_Flat` | 2.66 | [1.10, …] |

For all parameters `r_hat ≈ 1.00` and `n_eff` > 1700, indicating the chains converged.

**Conclusions:**
- on a small sample, the Bayesian model with non-informative priors performs almost identically to the classical one;
- informative priors noticeably improve ROC-AUC and recall, which matters in a medical setting (fewer missed patients);
- the Bayesian approach provides not only point estimates but full posterior distributions, i.e. an explicit measure of uncertainty for each coefficient.

## Installation and Usage

Requires Python 3.9+.

```bash
git clone <repository-url>
cd <project-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pyro-ppl torch pandas numpy scikit-learn statsmodels seaborn matplotlib jupyter
```

1. Place `processed_heart_data.csv` next to the notebook.
2. Run `jupyter notebook` and open `Large_dataset.ipynb`.
3. Execute the cells in order (**Run All**). MCMC sampling with 4 chains may take a few minutes.

The notebook also runs in Google Colab: upload the CSV to the session and run the cells.

## Repository Structure

```
.
├── Large_dataset.ipynb          # main analysis (classical + Bayesian LR)
├── processed_heart_data.csv     # preprocessed data
├── README.md
└── requirements.txt             # [recommended to add]
```

## Technologies

Python · PyTorch · Pyro (NUTS/MCMC) · statsmodels · scikit-learn · pandas · NumPy · seaborn · matplotlib

## Limitations and Possible Improvements

- the test set is small (46 observations), so metrics have high variance; consider adding confidence intervals or repeated splits;
- the first cross-validation in the notebook runs on the test portion (`X_test`) although the label says "Train Data" — this should be clarified or fixed;
- feature selection and final evaluation use the same split; it would be better to separate these stages;
- other priors could be compared (e.g. `Normal(0, 1)`, Laplace/horseshoe), along with posterior predictive checks (PPC);
- pin library versions in `requirements.txt`.

## Author

**[Name]** — [email / GitHub / LinkedIn]

## License

**[e.g. MIT]**. The data is distributed under the license of the original dataset.
