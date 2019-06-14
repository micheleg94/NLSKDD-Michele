# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:40:34 2019

@author: miche
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def ohe(train, test, categories):
    all_data = pd.concat((train,test))
    for column in all_data.select_dtypes(include=[np.object]).columns:
            train[column] = train[column].astype('category', categories = all_data[column].unique())
            test[column] = test[column].astype('category', categories = all_data[column].unique())
    
    for cat in categories:

            trainDum = pd.get_dummies(train[cat], prefix=cat)
            testDum = pd.get_dummies(test[cat], prefix=cat)
            train = pd.concat([train, trainDum.reindex(sorted(trainDum.columns),axis=1)],axis=1)
            test = pd.concat([test, testDum.reindex(sorted(testDum.columns), axis=1)],axis=1)
            train = train.drop(cat, axis=1)
            test = test.drop(cat, axis=1)
            
        
    return train, test


def standardScale(train, test, categories):
    scaler = StandardScaler()
    train[categories] = scaler.fit_transform(train[categories])  
    test[categories] = scaler.transform(test[categories])
    return train, test


def minMaxScale(train, test, categories):
    scaler = MinMaxScaler()
    train[categories] = scaler.fit_transform(train[categories])
    test[categories] = scaler.transform(test[categories])
    return train, test
    
def trainScaler(train, categories):
    scaler = StandardScaler()
    train[categories] = scaler.fit_transform(train[categories])  
    return train, scaler

def testScaler(test, categories, scaler):
    test[categories] = scaler.transform(test[categories])
    return test