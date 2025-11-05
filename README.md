# Titanic Survival Prediction - Kaggle Competition

Predict the survival of Titanic passengers using a clean, competition-ready machine learning pipeline.

## Overview
This project analyzes passenger attributes—such as age, gender, ticket class, fare, and family details—to predict survival chances in the Titanic disaster. The workflow covers data preprocessing, feature engineering, model training, cross-validation, and generation of a Kaggle-ready `submission.csv`.

## Key Features
- Robust preprocessing (missing values, encoding, scaling)
- Focused feature set: `Pclass`, `Sex`, `Embarked`, `LogAge`, `LogFare`, `FamilySize`, `IsAlone`, `Title`
- Machine Learning models used - XGBoost, Random Forest, Logistic Regression
- Stratified cross-validation for reliable accuracy estimates
- Generates `submission.csv` for Kaggle upload

## Project Structure
```
titanic-predictor/
├── LICENSE
├── test.csv                  # Dataset used for predicting
├── titanic_predictor.py      # Main script
├── train.csv                 # Dataset used for training
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

> ⚠️ Note: Do **not** commit Kaggle data files (`train.csv`, `test.csv`). Download them locally from the competition page and place them next to the script before running.

## Setup
1. **Clone the repo**
   ```bash
   git clone https://github.com/Mithi-07/titanic-survival-predictor.git
   cd titanic-survival-predictor
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the script:
   ```bash
   python titanic_predictor.py
   ```
2. After it runs, `submission.csv` file is saved in the same directory. It contains the passengers ID and the prediction 0 or 1 (Didn't survive or survived).

## Requirements
The minimal set is listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
xgboost
lightgbm
```
> Install with: `pip install -r requirements.txt`



## Acknowledgements
Dataset and competition by **Kaggle**: *Titanic - Machine Learning from Disaster*.
