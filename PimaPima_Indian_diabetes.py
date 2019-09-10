#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:37:05 2019

@author: pilaniya
"""

import numpy as np
import pandas as pd
dataset = pd.read_csv('Pima_Indian_diabetes.csv')

dataset.describe()
import matplotlib.pyplot as plt

"""dataset.plot(kind='Box',figsize=(15,10))
plt.show()""" 

dataset=dataset[dataset['Insulin']<600]
dataset=dataset[dataset['SkinThickness']<80]

"""dataset.plot(kind='Box',figsize=(15,10))
plt.show() """

"""dataset.loc[dataset['Glucose']==0,'Glucose']=dataset['Glucose'].mean()
dataset.loc[dataset['BloodPressure']==0,'BloodPressure']=dataset['BloodPressure'].mean()
dataset.loc[dataset['SkinThickness']==0,'SkinThickness']=dataset['SkinThickness'].mean()
dataset.loc[dataset['Insulin']==0,'Insulin']=dataset['Insulin'].mean()
dataset.loc[dataset['BMI']==0,'BMI']=dataset['BMI'].mean()"""

#dataset=dataset.dropna(axis=0,how='any')

"""dataset.describe()"""

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,8].values

from sklearn.impute import SimpleImputer
Imputer =SimpleImputer(missing_values=np.nan, strategy="mean",verbose=0)

Imputer=Imputer.fit(X[:,0:8])
X[:,0:8]=Imputer.transform(X[:,0:8])

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)



from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()

logr.fit(X_train,y_train)
prediction=logr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,prediction)

print(accuracy)
