# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

####Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####Veri yükleme
veriler = pd.read_csv('veriler.csv')
##veriler = pd.read_csv('c:\\Users\\e.cinar\\Documents\\GitHub\\PERSONAL\\MachineLearning\\veriler.csv')
#pd.read_csv("veriler.csv")
print(veriler)
####Veri ön işleme
boy =veriler [['boy']]
print(boy)

boykilo =veriler [['boy', "kilo"]]
print(boy)

class insan:
    boy = 180
    def kosmak(self, b):
        return b+10
    

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l = [1,3,5]