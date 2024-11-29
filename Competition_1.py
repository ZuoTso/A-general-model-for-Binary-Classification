# only for use google colab (drive)
# from google.colab import drive
# drive.mount('/content/drive')
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

dataset_names=[]
X_trains=[]
y_trains=[]
X_tests=[]
# drive_path = "./drive/MyDrive/Colab Notebooks/Competition_data" # only for use google colab (drive)
# for folder_name in os.listdir(drive_path):           # only for use google colab (drive)
for folder_name in os.listdir("./Competition_data"):
  # print(folder_name)
  dataset_names.append(folder_name)
  X_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/X_train.csv",header=0))
  y_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/y_train.csv",header=0))
  X_tests.append(pd.read_csv(f"./Competition_data/{folder_name}/X_test.csv",header=0))
#   X_trains.append(pd.read_csv(f"{drive_path}/{folder_name}/X_train.csv",header=0))  # only for use google colab (drive)
#   y_trains.append(pd.read_csv(f"{drive_path}/{folder_name}/y_train.csv",header=0))  # only for use google colab (drive)
#   X_tests.append(pd.read_csv(f"{drive_path}/{folder_name}/X_test.csv",header=0))   # only for use google colab (drive)
# print(y_trains)

## your code here
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

def select_best_model(X_train, y_train):
  """
  Multiple models and choose the best one
  based on AUC or cross-validation scores
  """
  models = {
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42)
  }

  best_model = None
  best_auc = 0
  best_model_name = ""

  X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
      X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
  )

  for model_name, model in models.items():
    model.fit(X_train_split, y_train_split)
    y_pred_prob = model.predict_proba(X_test_split)[:, 1]
    auc = roc_auc_score(y_test_split, y_pred_prob)

    if auc > best_auc:
      best_auc = auc
      best_model = model
      best_model_name = model_name

  best_model.fit(X_train, y_train)
  y_pred_prob = best_model.predict_proba(X_test_split)[:, 1]
  auc = roc_auc_score(y_test_split, y_pred_prob)
  
  return best_model, best_model_name, auc

# Main_1
models=[]
dataset_aucs=[]

for i in range(len(dataset_names)):
  tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(processed_X_trains[i], y_trains[i], test_size=0.2, random_state=42, stratify=y_trains[i])
  rf = RandomForestClassifier(random_state=42)
  param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
  }

  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
  grid_search.fit(tmp_X_train, tmp_y_train.squeeze())

  model = grid_search.best_estimator_
  models.append(model)

  tmp_y_prob = model.predict_proba(tmp_X_test)[:, 1]
  tmp_auc = roc_auc_score(tmp_y_test, tmp_y_prob)
  dataset_aucs.append(tmp_auc)

  # Now use select_best_model to see if there is a better model
  best_model, best_model_name, auc = select_best_model(processed_X_trains[i], y_trains[i])
  
  # Compare AUCs and update the model if needed
  if auc > tmp_auc:
    models[i] = best_model
    dataset_aucs[i] = auc

# Main_2
y_predicts = []
for i in range(len(dataset_names)):
  y_predict_proba = models[i].predict_proba(processed_X_tests[i])[:, 1]
  df = pd.DataFrame(y_predict_proba, columns=['y_predict_proba'])
  y_predicts.append(df)
for idx, dataset_name in enumerate(dataset_names):
  df = y_predicts[idx]
  df.to_csv(f'./Competition_data/{dataset_name}/y_predict.csv', index=False,header=True)
#   df.to_csv(f'{drive_path}/{dataset_name}/y_predict.csv', index=False, header=True)  # only for use google colab (drive)

# The fianl AUC
for idx, auc in enumerate(dataset_aucs):
  print(f"The fianl AUC of {dataset_names[idx]} : {auc:.4f}")
