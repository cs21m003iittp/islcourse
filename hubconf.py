import torch
import numpy as np
import sklearn
from sklearn.datasets import make_blobs,make_circles
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points)
  return X,y

def get_data_circles(n_points=100):
  pass
  X, y = make_circles(n_samples=n_points)
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
