# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:12:39 2019

@author: miche
"""

from sklearn.cluster import KMeans 
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
from clustering import clusteringKMeans

def reshapeFeature(x):
    feature=x.reshape(1,-1)
    
    return feature

def saveNpArray(X,Y):
    np.save(path + "\TrainX.npy", X)
    np.save(path + "\TrainY.npy", Y)
    
def loadNpArray():
    x = np.load(path+"\TrainX.npy")
    y = np.load(path+"\TrainY.npy")
    
    return x, y

def getXY(df):
    clssList = train.columns.values
    target = [i for i in clssList if i.startswith('classification')]
    Y=df[target]

    X = df.drop(target, axis=1)
    X = X.values
    
    return X, Y

def distance(X,dfNormal, dfAttack):
    X_Normal=np.array(dfNormal.drop(['classification'],1).astype(float))
    X_Attack=np.array(dfAttack.drop(['classification'],1).astype(float))
    
    centerAtt=clusteringKMeans(X_Attack, 50)
    centerNorm=clusteringKMeans(X_Normal, 60)

    matriceDist=[]

    for i in range(len(X)):
        feature1=reshapeFeature(X[i])
        dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
        dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
        row=[X[i],centerNorm[dist_matrixN[0].item()],centerAtt[dist_matrixA[0].item()]]
        matriceDist.append(row)

    return matriceDist


path = "D:\Tesi\dataset\DatasetAgg"
train = pd.read_csv(path+"\Train_encodedSample.csv", delimiter=",")

trainNormal = train[train['classification'] == 1]
trainAttack = train[train['classification'] == 0]

trainX, trainY = getXY(train)

dist = distance(trainX, trainNormal, trainAttack)
np.set_printoptions(suppress=True)
dist=np.array(dist)

saveNpArray(dist, trainY)
# =============================================================================
# matrice=[]
# for i in range(len(dist)):
#     matrice.append(dist[i])
# 
# =============================================================================
#file = open('matrix_Train111.csv', 'w')
#file.write(str(matrice[1]))






