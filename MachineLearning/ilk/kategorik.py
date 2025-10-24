# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

####Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
##veriler = pd.read_csv('c:\\Users\\e.cinar\\Documents\\GitHub\\PERSONAL\\MachineLearning\\veriler.csv')
#pd.read_csv("veriler.csv")
print(veriler)
#### Veri ön işleme
boy =veriler [['boy']]
print(boy)

boykilo =veriler [['boy', "kilo"]]
print(boykilo)

class insan:
    boy = 180
    def kosmak(self, b):
        return b+10
    

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

##### Eksik Veriler

#sci - kit lern
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
YAS = veriler.iloc[:,1:4].values
print(YAS)

"""
imputer = imputer.fit(YAS[:,0:3])
YAS[:,0:3]=imputer.transform(YAS[:,0:3])
"""
YAS = imputer.fit_transform(YAS)

print(YAS)


##### Kategorik verileri dönüştürme
from sklearn import preprocessing

ulke = veriler.iloc[:,0:1].values
print(ulke)

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


