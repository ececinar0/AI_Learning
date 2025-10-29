# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
print(veriler)

from sklearn import preprocessing
#tüm verileri tek seferde encod eder;
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
ohe = preprocessing.OneHotEncoder()

c = veriler2.iloc[:,:1]
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index=range(14),columns=['o','r','s'])
sonveriler = pd.concat([veriler2.iloc[:,-2:],havadurumu,veriler.iloc[:,1:3]], axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis = 1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float) # Tip dönüşümü

model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

"""
import statsmodels.api as sm #istatistiksel analiz ve özet (P-değerleri) sunan bir kütüphane

X_l = veri.iloc[:,[0,1,2,5]].values

#X = np.append(arr = np.ones((14,1)).astype(int), values = X_l, axis = 1) # tüm elemanları 1 olan 1 kolonlu dizi oluşturup, X_l'ye ekledik
X = np.array(X_l, dtype = float) # Tip dönüşümü


model = sm.OLS(ans,X).fit()
print(model.summary())

"""








