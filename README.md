# Credit Risk Scoring — Decision Trees & Ensemble Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6B23)
![Project](https://img.shields.io/badge/Project-Guided%20Learning-64748B)

A **guided machine learning project** developed while studying **Module 6 — Decision Trees and Ensemble Learning** of the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), created and taught by **Alexey Grigorev**.

The project explores how tree-based classification models can support **credit default risk assessment**, from data preparation and feature processing to model training, hyperparameter tuning, and evaluation.

> **Project attribution:** The problem, dataset, and core learning workflow come from ML Zoomcamp. This repository documents my hands-on implementation, experiments, and learning process. It is an educational exercise, not an original production credit-scoring system.

## 1. Business Problem

Financial institutions need to estimate the likelihood that an applicant will **default on a loan**. A predictive model can help identify patterns associated with credit risk and inform further review.

In this project, the objective is a **binary classification** task:

- **0 — Non-default:** the applicant is not labeled as a defaulter.
- **1 — Default:** the applicant is labeled as a defaulter.

The predicted probability represents the model's estimate of default risk. **It is not a standardized consumer credit score**, and it should not be interpreted as an automatic loan approval or rejection decision.

## 2. Dataset and Features

The project follows the **Credit Risk Scoring** example provided in [ML Zoomcamp Module 6](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/06-trees).

The data describes applicants and their loan requests. The modeling workflow includes the following input features:

| Feature type | Variables |
| --- | --- |
| Numerical | `seniority`, `time`, `age`, `expenses`, `income`, `assets`, `debt`, `amount`, `price` |
| Categorical | `home`, `marital`, `records`, `job` |
| Target | `status` (default vs. non-default) |

These variables cover employment tenure, applicant characteristics, financial position, loan conditions, housing, and credit history indicators.

**Data source:** See the official course materials linked above for the dataset and its original explanation. The notebook should be run with the dataset in the location expected by its data-loading code.

## 3. Machine Learning Workflow

### Data preparation

- Inspect and clean the dataset, including feature names and coded categorical values.
- Define the binary target and separate features from labels.
- Divide the data into training and validation/test subsets for model development and assessment.
- Keep preprocessing inside the modeling workflow to reduce the risk of data leakage.

### Feature preprocessing

A scikit-learn **`ColumnTransformer`** is used to apply different transformations to different types of data:

- **Numerical variables:** `StandardScaler`.
- **Categorical variables:** `OneHotEncoder(handle_unknown="ignore")`.

These transformations can be combined with an estimator in a single **`Pipeline`**, making preprocessing consistent during training, cross-validation, and prediction.

*Note: scaling is not usually necessary for tree-based models; it is included here as part of the implemented preprocessing pipeline.*

### Model development

Module 6 covers three main approaches to tree-based classification:

1. **Decision Trees** — interpretable rule-based models.
2. **Random Forest** — ensembles of trees trained on varied samples.
3. **XGBoost** — gradient-boosted trees that learn sequentially to improve predictions.

The implementation includes work with **`XGBClassifier`** inside a scikit-learn pipeline. Hyperparameter-search experiments use **`GridSearchCV`** to explore settings such as:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_child_weight`

When tuning an estimator inside a pipeline, parameter names must be prefixed with the estimator step name (for example, `xgbclassifier__max_depth`).

### Evaluation and model selection

**ROC-AUC** is used as a key metric for assessing how well a classifier ranks applicants by default risk. Cross-validation helps compare candidate parameter settings beyond one training/validation split.

Model selection should consider not only predictive performance, but also validation stability, overfitting, and computational cost. Refer to the notebooks for the actual experiments and their recorded results; no unverified performance figures are claimed here.

## 4. Tools and Technologies

| Tool | Purpose |
| --- | --- |
| Python | Data analysis and model development |
| pandas / NumPy | Data manipulation and numerical operations |
| scikit-learn | Preprocessing, pipelines, model selection, and evaluation |
| XGBoost | Gradient-boosted classification |
| Jupyter | Interactive experiments and documentation |

## 5. Running the Project Locally

**Prerequisites:** Python 3 and Git. A Python environment compatible with the notebook's dependencies is recommended.

1. Clone or download this repository, then open a terminal in the project folder.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   **Windows (PowerShell):**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. Install the main packages if you do not already have a project-specific dependency file:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyterlab
   ```

5. Make sure the dataset is available at the path used by the notebook. Launch Jupyter:

   ```bash
   jupyter lab
   ```

6. Open the project's credit-risk-scoring notebook and run its cells in sequence.

> **Reproducibility note:** Dependency versions, the dataset path, random seeds, and any final metrics should be taken from the actual project environment/notebook. The commands above describe the general workflow rather than a verified locked environment.

## 6. Key Learning Outcomes

This exercise helped me practice:

- Framing credit default prediction as a binary classification problem.
- Handling numerical and categorical features in a unified preprocessing pipeline.
- Understanding how decision trees, bagging, and gradient boosting differ.
- Using XGBoost with scikit-learn estimators and pipelines.
- Performing hyperparameter search with cross-validation.
- Evaluating models using ROC-AUC and considering overfitting.
- Making experiments more structured and reproducible.

## 7. Limitations and Possible Improvements

This is an **educational project**, not a model approved for real financial decision-making. A real-world system would require additional work, including:

- Calibration and assessment of predicted probabilities.
- Precision/recall and threshold analysis for different business costs.
- Bias and fairness testing, explainability, and appropriate governance.
- Robust evaluation on out-of-time and out-of-sample data.
- Monitoring for data drift and changes in model performance.
- Model versioning, reproducible dependency management, and deployment tests.

## 8. Acknowledgements

This work is based on the **[Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)** by **[DataTalks.Club](https://datatalks.club/)**, taught by **Alexey Grigorev**.

- [Module 6 — Decision Trees and Ensemble Learning](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/06-trees)
- [ML Zoomcamp course overview](https://github.com/DataTalksClub/machine-learning-zoomcamp)

Thanks to the course creators and community for making practical ML engineering education openly available.

---

**Repository owner:** [lohengrin-01](https://github.com/lohengrin-01)  
**Project type:** Guided coursework / hands-on machine learning practice
