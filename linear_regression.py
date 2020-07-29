# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:20:16 2020

@author: RJ PC
"""

import pandas as pd
df = pd.read_csv(r"X:\ML\algos\Linear_regression\Multiple-Linear-Regression-master\50_Startups.csv")

df.head()

x = df.iloc[:,:-1]
y = df.iloc[:,4]

states = pd.get_dummies(x['State'], drop_first=True)

x = x.drop('State', axis=1)

x = pd.concat([x,states], axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.10, random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression().fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_pred,y_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)*100
