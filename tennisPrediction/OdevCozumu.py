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
#pd.read_csv("veriler.csv")
#test
#print(veriler)
#veri on isleme

from sklearn import preprocessing
##tüm verileri tek seferde encod eder;
#veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

le= preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

"""
outlook = veriler.iloc[:,:1].values
#weather = veriler.iloc[:,1:3].values
windy = veriler.iloc[:,3:-1].values
ans = veriler.iloc[:,4:].values
"""

"""
outlook = le.fit_transform(veriler.iloc[:,0])
windy = le.fit_transform(veriler.iloc[:,3])
ans = le.fit_transform(veriler.iloc[:,4])
"""
"""
outlook = ohe.fit_transform(veriler.iloc[:,:1]).toarray()
windy = ohe.fit_transform(veriler.iloc[:,3:-1]).toarray()
ans = ohe.fit_transform(veriler.iloc[:,4:]).toarray()


part1 = pd.DataFrame(data=outlook, index = range(14),columns = ['overcast','rainy','sunny'])
part2 = pd.DataFrame(data=weather, index = range(14),columns = ['temp','humid'])
part3 = pd.DataFrame(data=windy, index = range(14),columns = ['windy false','windy true'])
ans = pd.DataFrame(data=ans, index = range(14),columns = ["can't","can"])
"""
weather = veriler.iloc[:,1:3].values #sayısal verileri dönüştürmeye gerek duymadım
outlook = ohe.fit_transform(veriler.iloc[:,:1]).toarray() #sıralama olmayan kategorik veriler olduğu için One Hot Encoder kullandım
windy = le.fit_transform(veriler.iloc[:,3]) #iki kategori olduğunda dummy etkisi olmaması için Label Encoder kullandım
ans = le.fit_transform(veriler.iloc[:,4])#Label Encoder 0,1,2 şeklinde numaralandırır. 
#2 kategori olduğunda birinin 0 olması diğerinin 1 olması demektir. Bu yüzden Label Encoder yeterlidir.
#Sonuç olarak, kategorik verilerde 2 kategori için Label Encoder yeterlidir. 
#3 ve daha fazla kategori için One Holt Encoder kullanılır. 

part1 = pd.DataFrame(data=outlook, index = range(14),columns = ['overcast','rainy','sunny'])
part2 = pd.DataFrame(data=weather, index = range(14),columns = ['temp','humid'])
part3 = pd.DataFrame(data=windy, index = range(14),columns = ['windy'])
ans = pd.DataFrame(data=ans, index = range(14),columns = ["ans"])

veri = pd.concat([part1,part2,part3], axis = 1)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(veri,ans,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


"""
import statsmodels.api as sm #istatistiksel analiz ve özet (P-değerleri) sunan bir kütüphane

X_l = veri.iloc[:,[0,1,2,5]].values

#X = np.append(arr = np.ones((14,1)).astype(int), values = X_l, axis = 1) # tüm elemanları 1 olan 1 kolonlu dizi oluşturup, X_l'ye ekledik
X = np.array(X_l, dtype = float) # Tip dönüşümü


model = sm.OLS(ans,X).fit()
print(model.summary())

"""









################  REFERANS KODLAR  ################

"""
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

'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


boy = s2.iloc[:,3:4].values
c = s2.iloc[:,-1:].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis = 1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)


#### BACKWARD AL
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(c,X_l).fit()
print(model.summary())



"""























