# Employee Turnover Prediction: HR Analytics

## Project Overview
This project applies machine learning to HR analytics to predict employee turnover. By analyzing a dataset of nearly 15,000 employees, the goal is to identify the key drivers of attrition and provide actionable insights for retention strategies. 

The primary model used in this project is **Logistic Regression**, chosen for its interpretability and ability to output log-odds, allowing us to understand exactly how much each feature influences an employee's decision to leave.

## Dataset
* **Source:** HR Analytics Dataset (`HR_comma_sep.csv`)
* **Size:** 14,999 records
* **Target Variable:** `left` (0 = Stayed, 1 = Left) - Baseline turnover rate is 23.8%
* **Features:** Satisfaction level, last evaluation score, number of projects, average monthly hours, tenure, work accidents, promotions, department, and salary.

## Methodology & Pipeline
To prepare the data for the Logistic Regression model, the following preprocessing steps were applied:
1. **One-Hot Encoding:** Applied to categorical variables (`Department` and `Salary`) to prevent the model from assigning false ordinal weight to nominal categories.
2. **Feature Scaling:** Applied `StandardScaler` to continuous numerical features (e.g., `average_montly_hours`, `satisfaction_level`) to ensure uniform coefficient weighting and model convergence.
3. **Evaluation Metrics:** Due to the class imbalance (23.8% turnover), the model was evaluated using **ROC-AUC**, **Precision**, and **Recall** rather than raw accuracy.

## Repository Structure
```text
├── data/                   # Contains the HR_comma_sep.csv dataset
├── notebooks/              # Jupyter notebooks for EDA and model training
├── src/                    # Python scripts for data processing
├── README.md               # Project documentation
├── requirements.txt        # Package dependencies
└── .gitignore              # Ignored files and directories
