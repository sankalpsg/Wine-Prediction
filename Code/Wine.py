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

#Missing Values
  Missing_Values = white_wine.isnull().sum()
  print(Missing_Values)

# Check If Dataset is balanced
  balance = white_wine['quality'].value_counts()
  print(balance)
  
  vc = white_wine['quality'].value_counts()
  vc = vc.sort_index()
  vc.plot(kind='bar',use_index=True,legend=True,rot=0,x='Quality' ,y='Count',title='Distribution of Quality')
  objects =[0,1,2,3,4,5,6,7,8,9,10]
  y_pos=np.arange(len(objects))
  plt.bar(y_pos, white_wine['quality'],legends=True ,align='center', alpha=0.5)
  plt.xticks(y_pos, objects)



# We see the dataset is imbalanced,we will use SVM algorithm with penalization to represent the imbalance.Also
# Decision trees  perform well on imbalanced datasets because their 
# hierarchical structure allows them to learn signals from both classes.






# For the feature selection, we remove highly correlated features to avoid collinearity to find key 
# features that affect wine quality. We also remove features that have a negligible effect on quality.
#A clear relationship between quality and the features can be obtained through correlation matrices.

# Correlation Matrix

  import seaborn as sns
  sns.set(style= "white")
  f, ax = plt.subplots(figsize=(10, 8))
  corr = white_wine.corr()
  sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
              square=True, ax=ax)
  
# Variance Table
  
  Var = np.var(X)
  print(Var)
  
# From the matrix, alcohol is the most important driver of quality with 0.435. Density and 
# Volatile Acidity are the next two features which exhibit strongest correlations with quality.
# But as density highly correlates with alcohol, we will not consider it to avoid collinearity.
# Also density has near to zero variance to impact the ratings of white wine.
# Total Sulphur dioxide has the most variance with 1805 and has good coorelation with quality 
# to be considered.


# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',class_weight='balanced', # penalize
                 probability=True,random_state = 0)
classifier.fit( X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y= y_train, cv= 10)
accuracies.mean()
accuracies.std()



#Feature Selection using Scilearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])






if __name__ == '__main__':
  main()

