# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

####  1. Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####  2. Veri Önişleme

####  2.1. Veri Yükleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)
"""
#veriler = pd.read_csv('c:\\Users\\e.cinar\\Documents\\GitHub\\PERSONAL\\MachineLearning\\eksikveriler.csv')
#veriler = pd.read_csv("eksikveriler.csv")
"""

#### 2.1.1. Veri ön işleme
boy = veriler [['boy']]
print(boy)

boykilo =veriler [['boy', "kilo"]]
print(boykilo)


##### 2.2. Eksik Veriler

##### 2.2.1. eksik verileri doldurmak (Impute etmek)
#sci - kit lern
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')# strategy='median' or strategy='most_frequent'
YAS = veriler.iloc[:,1:4].values
print(YAS)

"""
imputer = imputer.fit(YAS[:,0:3])
YAS[:,0:3]=imputer.transform(YAS[:,0:3])
"""
YAS = imputer.fit_transform(YAS)

print(YAS)

##### 2.3. Encoding işlemi (Kategorik -> numeric)
from sklearn import preprocessing

ulke = veriler.iloc[:,0:1].values
print(ulke)

##### 2.3.1. Her bir değere sayısal değer atama
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

##### 2.3.2. Her bir değer için kolon oluşturup o kolona isim verme ve değeri belirtme  
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


##### 2.4 Verileri birleştirme (concat)

##### 2.4.1 numpy dizileri dataFrame dönüşümü
sonuc1 = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc1)

sonuc2 = pd.DataFrame(data = YAS, index = range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)



##### 2.4.2 dataFrame birleştirme işlemi
"""
## Tek seferde birleştirme

s = pd.concat([sonuc1,sonuc2,sonuc3],axis=1)
print(s)
"""
s = pd.concat([sonuc1,sonuc2], axis=1 )
#print(s)

s2 = pd.concat([s,sonuc3], axis=1 )
print(s2)



##### 2.5. Verilerin Eğitimi

##### 2.5.1. 'Giriş ve sonuç'(dikeyde) , 'Eğitim ve Test'(yatayda) olarak bölme

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)



##### 2.5.2. Öznitelik Ölçekleme
### Birbirlerine yakın değerler elde ediyorlar, aynı dünyaya getirmek deniyor

from sklearn.preprocessing import StandardScaler  #normalization da yapılabilir

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)



























