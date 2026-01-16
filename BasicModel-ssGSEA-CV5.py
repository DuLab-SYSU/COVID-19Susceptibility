#!/usr/bin/env python
# coding: utf-8

# Build an ssGSEA-based baseline model

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Load dataset
data1 = pd.read_csv('ssgsea_10.csv')
X = data1.iloc[:, 1:]
y = data1.iloc[:, 0] 

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Objective function
def objective(trial, model_name):
    elif model_name == 'logistic':
        model = LogisticRegression(
            C=trial.suggest_float('C', 1e-4, 1e2, log=True),
            penalty='l1',
            solver='saga',
            random_state=42
        )

    elif model_name == 'gnb':
        model = GaussianNB(
            var_smoothing=trial.suggest_float('var_smoothing', 1e-12, 1e-7, log=True)
        )

    # Compute mean AUC using 5-fold cross-validation
    auc_scores = []
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred))

    return np.mean(auc_scores)

# Run Optuna for hyperparameter optimization
optimized_models = {}
best_aucs = {}

for model_name in ['logistic', 'gnb']:
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, model_name), n_trials=50)

    optimized_models[model_name] = study.best_params
    best_aucs[model_name] = study.best_value

    print(f"Best AUC for {model_name}: {study.best_value}")
    print(f"Best parameters for {model_name}: {study.best_params}")

# Output the best AUC for each model
print("\nModel Best AUCs:")
for model_name, auc in best_aucs.items():
    print(f"{model_name}: {auc}")

import joblib
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Initialize models using the best parameters
optimized_models_instances = {
    'logistic': LogisticRegression(**optimized_models['logistic'], random_state=42),
    'gnb': GaussianNB(**optimized_models['gnb']),
}

# Save each optimized model
for model_name, model in optimized_models_instances.items():
    joblib.dump(model, f'./basemodel/ssgsea/{model_name}_ssgsea.joblib')
    print(f"Model {model_name} saved successfully.")
