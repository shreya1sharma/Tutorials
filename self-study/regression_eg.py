# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:30:54 2017

@author: hoge
"""
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] *100.00
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] *100.00

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_column= 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label']= df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)

X= np.array(df.drop(['label'],1))
y= np.array(df['label'])
X= preprocessing.scale(X)
#X= X[:-forecast_out+1]
#df.dropna(inplace=True)
y= np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf =LinearRegression()
clf= LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)

confidence= clf.score(X_test, y_test)