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
 
def build_lr_model(X, y):
  
  lr_model = LogisticRegression().fit(X, y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X, y):
  
  
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model=RandomForestClassifier()
  rf_model.fit(X, y)
  
  return rf_model

def get_metrics(model,X,y):
  
  y_test=y
  y_pred_test = model.predict(X)
  
  
  
  
  
  # View accuracy score
  acc=accuracy_score(y_test, y_pred_test)
  # print(acc)
  rec=recall_score(y_test,y_pred_test,average='macro')
  #print(rec)
  prec=precision_score(y_test,y_pred_test,average='macro')
  #print(prec)
  f1=f1_score(y_test,y_pred_test,average='macro')
  
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test, pos_label=2)
  auc = metrics.auc(fpr, tpr)
  
  #auc=roc_auc_score(y_test,y_pred_test,multi_class='ovr')
  #print(auc)
  # write your code here...
  
  return acc, prec, rec, f1, auc
 
