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


#veri on isleme
ulke = veriler.iloc[:,0:1].values #OHE işlemlerinde bu değişkenleri kullandığımız için gerekli ama sütunu OHE içerisinde belirtirsek gerek kalmaz.
cinsiyet = veriler.iloc[:,-1:].values #OHE işlemlerinde bu değişkenleri kullandığımız için gerekli ama sütunu OHE içerisinde belirtirsek gerek kalmaz.
Yas = veriler.iloc[:,1:4].values #Zaten sayısal veri olduğu için dönüşüm yapılmayacak ama birleştirme işlemi için tanımlıyoruz
#print(ulke)
 
from sklearn import preprocessing
le= preprocessing.LabelEncoder() #Kategori numarası verir
ohe = preprocessing.OneHotEncoder() #Her kategori için 0/1 lı sütunlar oluşturur

#ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)
#verilerin egitim ve test icin bolunmesi

"""

***ulke[:,0] = le.fit_transform(...) bu satırda ulke[:,0] atanan değişkende [:,0] tanımlamasını anlamadım


Bu, NumPy dizilerinde (array) veri seçme ve atama işlemleri için kullanılan "dilimleme" (slicing) sözdizimidir (syntax).

ulke[:,0] ifadesini parçalara ayıralım:

ulke: Bu, sizin ulke = veriler.iloc[:,0:1].values satırınızla oluşturduğunuz bir NumPy dizisidir.

Önemli nokta: veriler.iloc[:,0:1] (sonda :1 kullandığınız için) size 1 sütunlu bir DataFrame verir.

Bunun .values özelliğini aldığınızda, ulke değişkeni (22, 1) şeklinde 2 boyutlu bir NumPy dizisi olur (22 satır, 1 sütun).

[...]: NumPy'de indeksleme/dilimleme bu parantezlerle yapılır.

:,0: Bu, "satırlar" ve "sütunlar" için verilen bir komuttur ve virgülle ayrılır: [satır_seçimi, sütun_seçimi]

: (Virgülden önce): "Tüm satırları seç" anlamına gelir.

0 (Virgülden sonra): "Sadece 0. indeksteki sütunu seç" anlamına gelir.


    ÖRNEK OLARAK BU ŞEKİLDE YAPILABİLİR::
    
    
        veriler2 = pd.read_csv('veriler.csv')
        veriler2 = veriler2.iloc[:,:].values #NumPy dizisi olur
        veriler2[:,-1] = le.fit_transform(veriler.iloc[:,-1]) #Son kolonu 0/1 yapar


            PRINT: son kolon 0/1'lere dönüştürüldü.
"""

sonuc = pd.DataFrame(data=ulke, index = range(22),columns = ['fr','tr','us']) #Bu sayısal dizileri tekrar etiketli DataFrame'lere dönüştürür.
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas']) #Bu sayısal dizileri tekrar etiketli DataFrame'lere dönüştürür.
print(sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet[:,-1:], index = range(22), columns = ['cins']) 
#OHE ile 2 kolonlu bir dizi olmuştur. # data = cinsiyet[:,-1:] ---> (Dummy etkisi olmaması için) 1 kolonu seçer
print(sonuc3) 

s = pd.concat([sonuc,sonuc2],axis = 1)
s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)


from sklearn.model_selection import train_test_split

## ÖĞRENME VE TEST için verinin giriş(X) çıkış(Y) olarak ayrılması işlemleri
# öğrenme giriş/çıkış ve test giriş/çıkış olmak üzere 4 dataFrame'e ayrılması
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
regressor.fit(x_train,y_train) #regresyon doğrusu oluşturur

y_pred = regressor.predict(x_test)


boy = s2.iloc[:,3:4].values
c = s2.iloc[:,-1:].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis = 1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)
#test_size=0.33 ---> test verisi için %33lük veriyi ayırır.
#random_state=0 --->

r2 = LinearRegression()
r2.fit(x_train,y_train)#Modeli eğitir. Bu, geçerli bir Lineer Regresyon görevidir.

y_pred = r2.predict(x_test)


#### BACKWARD AL
## "tüm özellikler gerçekten gerekli mi, yoksa bazıları gereksiz mi?" sorusunu istatistiksel olarak yanıtlamaya çalışır
# Amaç tahminde "anlamsız" (P-değeri > 0.05) değeri bulmak
import statsmodels.api as sm #istatistiksel analiz ve özet (P-değerleri) sunan bir kütüphane

X_l = veri.iloc[:,[3,5]].values ##ülke sütunlarının toplamı 1 olduğu için birini almadık.  X ile kukla etkisi yaratıyor.
#yukarıda dummy etkisini engelleme işlemine k-1 kuralı deniyormuş

X = np.append(arr = np.ones((22,1)).astype(int), values = X_l, axis = 1) # tüm elemanları 1 olan 1 kolonlu dizi oluşturup, X_l'ye ekledik
X = np.array(X, dtype = float) # Tip dönüşümü


model = sm.OLS(boy,X).fit()
"""
1) Modelin tanımlanması: sm.OLS(boy, X) --->(OLS) Ordinary Least Squares" (En Küçük Kareler Yöntemi). 
    lineer regresyon modelini oluşturmak için kullanılan standart matematiksel yöntem
        "Ben bir 'En Küçük Kareler' (Lineer Regresyon) modeli kurmak istiyorum. Amacım boy'u tahmin etmek. 
        Bunu yaparken X'in içindeki tüm sütunları kullan."
2) Modelin eğitilmesi (hesaplanması): .fit()
    "En Küçük Kareler" yöntemini kullanarak bu verilere uyan lineer regresyon formülünün katsayılarını 
    ($\beta$ değerlerini) hesaplar:
        boy = \beta_0 \times (sabit) + \beta_1 \times (tr) + \beta_2 \times (us) + \beta_3 \times (kilo) + ...
    P-değerleri, R-kare, katsayılar, standart hatalar vb.) yapar ve bu sonuçları içeren bir "sonuç paketi" (results object) oluşturur
    
3) Sonuçların atanması: model = ...
"""

print(model.summary())

























