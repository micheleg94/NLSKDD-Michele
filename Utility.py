# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:50:57 2019

@author: miche
"""
import numpy as np

def reshapeFeature(x):
    feature=x.reshape(1,-1)
    
    return feature

def saveNpArray(X,Y, tipo):
    filenameX = tipo + "X.npy"
    filenameY = tipo + "Y.npy"
    np.save(path + filenameX, X)
    np.save(path + filenameY, Y)
    
def loadNpArray():
    x = np.load(path+"\Train\FullTrainX.npy")
    y = np.load(path+"\Train\FullTrainY.npy")
    
    return x, y

def getXY(df):
    target = ['classification']
    Y=df[target]

    X = df.drop(target, axis=1)
    X = X.values
    
    return X, Y

path = "D:\Tesi\dataset\DatasetAgg"

