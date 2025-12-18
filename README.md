# Insurance Risk Classification Project

## Project Overview

This project focuses on building a machine learning classification model to identify **high-risk insurance members** based on demographic, lifestyle, medical, and policy-related attributes. Accurately identifying high-risk individuals enables insurance providers to proactively manage care, optimize premiums, and reduce long-term medical costs.

---

## Business Problem

Insurance providers face rising medical costs driven by a small proportion of high-risk members. Failing to identify these individuals early can result in unmanaged care and significant financial losses. The challenge is to develop a predictive model that can reliably classify members as **high risk** or **not high risk**, allowing targeted interventions and efficient resource allocation.

---

## Business Objective

The primary objective is to build a **classification model** that:

* Accurately identifies high-risk individuals
* Minimizes **false negatives** (missed high-risk members)
* Provides interpretable insights to support business decision-making

Given the cost implications, **recall for high-risk individuals** is prioritized over overall accuracy.

---

## Dataset Description

The dataset contains anonymized insurance member information, including:

* **Demographics:** age, sex, region, marital status, household size
* **Socioeconomic factors:** income, education, employment status
* **Lifestyle indicators:** smoking status, alcohol frequency, BMI
* **Medical history:** chronic conditions, hospitalizations, lab results
* **Insurance details:** plan type, premiums, deductibles, provider quality
* **Target variable:** `is_high_risk` (1 = High Risk, 0 = Not High Risk)

---

## Exploratory Data Analysis (EDA)

Key insights from EDA include:

* The dataset is moderately imbalanced, with fewer high-risk members than low-risk members
* High-risk individuals tend to have more chronic conditions, higher hospital utilization, and worse clinical indicators
* Lifestyle factors such as smoking and elevated BMI are strongly associated with risk status

EDA helped identify important predictors and potential data leakage risks.

---

## Data Preprocessing

Preprocessing steps included:

* Handling missing values
* Encoding categorical variables using **dummy (one-hot) encoding**
* Feature scaling for models sensitive to feature magnitude
* Removing **leaky features** derived from post-outcome information (e.g., total claims paid)

These steps ensured model validity and prevented target leakage.

---

## Modeling Approach

Two classification models were developed and evaluated:

### 1. Logistic Regression (Baseline Model)

* Provides strong interpretability
* Achieved high accuracy and recall after addressing data leakage
* Served as a transparent benchmark model

### 2. Random Forest Classifier

* Captures non-linear relationships
* Achieved superior performance metrics
* Reduced false negatives compared to Logistic Regression

---

## Model Evaluation

Models were evaluated using:

* Confusion Matrix
* Precision, Recall, and F1-score
* **ROC–AUC** for threshold-independent performance comparison

### ROC–AUC Results

* Logistic Regression: **0.9976**
* Random Forest: **0.9997**

Both models demonstrated outstanding discriminatory power, with Random Forest performing marginally better.

---

## Threshold Tuning

The default classification threshold (0.5) was adjusted to better align with business priorities. Lowering the threshold:

* Increased recall for high-risk individuals
* Reduced the likelihood of missing costly cases
* Introduced a manageable increase in false positives

A tuned threshold provided a better balance between business risk and operational efficiency.

---

## Final Recommendation

The **Random Forest model with a tuned decision threshold** is recommended for deployment due to:

* Near-perfect ROC–AUC performance
* High recall for high-risk individuals
* Significant reduction in missed high-risk cases

Logistic Regression remains a strong alternative when model interpretability is the primary concern.

---

## Key Takeaways

* Data leakage detection and correction is critical for reliable modeling
* Business objectives should guide metric selection and threshold tuning
* Advanced models can improve performance, but interpretability must be considered

---

## Future Work

Potential improvements include:

* Cost-sensitive learning
* Model monitoring and drift detection
* Integration into a real-time decision support system
* Explainability techniques such as SHAP values

---

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## Author

**Vincent Muteti**

Data Science & Machine Learning Project
