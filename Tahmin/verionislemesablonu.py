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
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

ulke = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,-1:].values
Yas = veriler.iloc[:,1:4].values
#print(ulke)

from sklearn import preprocessing
le= preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)
#verilerin egitim ve test icin bolunmesi

sonuc = pd.DataFrame(data=ulke, index = range(22),columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet[:,-1:], index = range(22), columns = ['cins'])
print(sonuc3)

s = pd.concat([sonuc,sonuc2],axis = 1)
s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)










