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


def main():
 
# Importing the dataset
  white_wine = pd.read_csv('winequality-white.csv',sep=';')
  
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

#Missing Values
  Missing_Values = white_wine.isnull().sum()
  print(Missing_Values)

# Check If Dataset is balanced
 balance = white_wine['quality'].value_counts()
 print(balance)
 plt.hist(white_wine['quality'])
 plt.title("Frequency Table")
 plt.xlabel("Quality")
 plt.ylabel("Frequency")




if __name__ == '__main__':
  main()
