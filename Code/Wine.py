# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:33:54 2017

@author: Sankalp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("C:/Users/Sankalp/Desktop/DataScience Trinity/Machine Learning/Assignment/Assignment 3")


def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale)

def add_categ_quality(y):
    low = y[y['quality'] <= 5]
    medium = y[(y['quality'] == 6) | (y['quality'] == 7)]
    high = y[y['quality'] > 7]

    low['quality_mark'] = 'low'
    medium['quality_mark'] = 'medium'
    high['quality_mark'] = 'high'

    frames = [low, medium, high]
    return pd.concat(frames)


def main():
 
# Importing the dataset
  white_wine = pd.read_csv('winequality-white.csv',sep=';')
  white_wine= add_categ_quality(white_wine)
  white_wine.describe().transpose()

# Independent and Target Variables
  X = white_wine.iloc[:, 0:11].values
  y = white_wine.iloc[:,11].values

# Normalization of Data
# rescale training data to lie between 0 and 1
  X = normaliseData(X)
   
#Training and Test Data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
#Applying PCA Analysis
  from sklearn.decomposition import PCA
  pca = PCA(n_components = None)
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  Variance_Explained = pca.explained_variance_ratio_
  print(Variance_Explained)

issing Values
  

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',class_weight='balanced', # penalize
                 probability=True,random_state = 0)
classifier.fit( X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Cross Validation






if __name__ == '__main__':
  main()

