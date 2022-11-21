import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,recall_score,roc_auc_score,precision_score,f1_score
from sklearn.model_selection import GridSearchCV
import sklearn
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection

 #Part1

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points)
  return X,y

def get_data_circles(n_points=100):
  pass
  X, y = make_circles(n_samples=n_points)
  return X,y

def get_data_mnist():
  digits= load_digits()
  X = digits.data
  y = digits.target
  
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  
  km = KMeans(n_clusters=k)
  km.fit(X)
  return km


def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1,ypred_2):
  pass
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v

 
 #Part 2
 
def build_lr_model(X=None, y=None):
  # write your code...
  # Build logistic regression, refer to sklearn
  scaler = preprocessing.StandardScaler().fit(X)
  X_train = scaler.transform(X)
  lr_model = None
  lr_model = LogisticRegression(random_state=0).fit(X_train, y)
  return lr_model

def build_rf_model(X=None, y=None):
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model = None
  rf_model = RandomForestClassifier(random_state=0, max_depth=5).fit(X, y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  ypred = model1.predict(X)

  acc = metrics.accuracy_score(y, ypred)
  prec = metrics.precision_score(y, ypred, average='macro')
  rec = metrics.recall_score(y, ypred, average='macro')
  f1 = metrics.f1_score(y, ypred, average='macro')
  fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=2)
  auc = metrics.auc(fpr, tpr)
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = None
  lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    # 'C': [1.0, 2.0],
    'solver': ['liblinear'],

  }
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = None
  rf_param_grid = {
    'n_estimators': list(range(10,101,10)),
    'max_features': list(range(6,32,5)),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 5, 10, None]
  }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc_ovo', 'roc_auc_ovr']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  grid_search_cv = None
  grid_search_cv = model_selection.GridSearchCV(model1, param_grid, cv=cv, scoring=metrics, refit=False).fit(X, y)
  cv_results = grid_search_cv.cv_results_
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation

  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  top1_scores = []

  for metric in metrics:
    i = list(cv_results[f'rank_test_{metric}']).index(1)
    top_score = list(cv_results[f'mean_test_{metric}'])[i]
    top1_scores.append(top_score)
  
  return top1_scores
