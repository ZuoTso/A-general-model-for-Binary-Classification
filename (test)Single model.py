import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

dataset_names=[]
X_trains=[]
y_trains=[]
X_tests=[]
for folder_name in os.listdir("./Competition_data"):
  if folder_name == "Dataset_15":
    dataset_names.append(folder_name)
    X_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/X_train.csv",header=0))
    y_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/y_train.csv",header=0))
    X_tests.append(pd.read_csv(f"./Competition_data/{folder_name}/X_test.csv",header=0))

def preprocess_data(X_train, X_test):
  """
  Data Preprocessing & Feature Engineering
    a) Automatically identify numerical and categorical features
    b) Feature Scaling
    c) Categorical features
  """
  # a
  numeric_features = X_train.select_dtypes(include=['float64']).columns
  categorical_features = X_train.select_dtypes(include=['int64']).columns

  # b
  scaler = StandardScaler()

  if len(numeric_features) > 0:  # if numerical features
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

  # c

  return X_train, X_test

processed_X_trains = []
processed_X_tests = []

for i in range(len(dataset_names)):
  X_train, X_test = X_trains[i], X_tests[i]
  X_train, X_test = preprocess_data(X_train, X_test)
  processed_X_trains.append(X_train)
  processed_X_tests.append(X_test)
  print(processed_X_trains)
  print(processed_X_tests)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
      X_train, y_trains[0], test_size=0.2, random_state=42, stratify=y_trains[0]
  )

# Good
# model = SVC(kernel='linear', class_weight='balanced', probability=True)
model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

# Bad
# model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

train_accuracy = model.score(X_train_split, y_train_split)
test_accuracy = model.score(X_test_split, y_test_split)

y_pred_prob = model.predict_proba(X_test_split)[:, 1]
auc = roc_auc_score(y_test_split, y_pred_prob)

print(f"AUC Score: {auc:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
