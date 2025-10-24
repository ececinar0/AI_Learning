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


##### Verileri birleştirme (concat)
print(list(range(22)))

sonuc1 = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc1)

sonuc2 = pd.DataFrame(data = YAS, index = range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)

"""
## Tek seferde birleştirme

s = pd.concat([sonuc1,sonuc2,sonuc3],axis=1)
print(s)
"""
s = pd.concat([sonuc1,sonuc2], axis=1 )
#print(s)

s2 = pd.concat([s,sonuc3], axis=1 )
print(s2)


##### 'Giriş ve sonuç'(dikeyde) , 'Eğitim ve Test'(yatayda) olarak bölme

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)