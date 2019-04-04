# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:30:01 2019

@author: miche
"""


from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
import time
from Utility import reshapeFeature, saveNpArray,getXY



def findNearestTrain(df,dfNormal, dfAttack):
    dfNormal1=dfNormal.drop(['classification'],1)
    dfAttack1=dfAttack.drop(['classification'],1)
    X_Normal=np.array(dfNormal.drop(['classification'],1).astype(float))
    X_Attack=np.array(dfAttack.drop(['classification'],1).astype(float))

    matriceDist=[]
    for i in range(len(X_Normal)):
        
        df2=dfNormal1._slice(slice(i, i+1))
        df2=np.array(df2).astype(float)
        feature1=reshapeFeature(df2[0])
        X_Normal[i,:]=0
        
        dist_matrixN = pairwise_distances_argmin_min(feature1,X_Normal)
        dist_matrixA = pairwise_distances_argmin_min(feature1,X_Attack)
        row=[X_Normal[i],X_Normal[dist_matrixN[0].item()],X_Attack[dist_matrixA[0].item()]]
        matriceDist.append(row)
        X_Normal[i]=feature1
           
    for i in range(len(X_Attack)):
        df2=dfAttack1._slice(slice(i, i+1))
        df2=np.array(df2).astype(float)
        feature1=reshapeFeature(df2[0])
        X_Attack[i,:]=0
        
        dist_matrixN = pairwise_distances_argmin_min(feature1,X_Normal)
        dist_matrixA = pairwise_distances_argmin_min(feature1,X_Attack)
        row=[X_Attack[i],X_Normal[dist_matrixN[0].item()],X_Attack[dist_matrixA[0].item()]]
        matriceDist.append(row)
        X_Attack[i]=feature1  
    return matriceDist


def findNearestTest(X_test,dfNormal,dfAttack):
    XNormal=np.array(dfNormal.drop(['classification'],1).astype(float))
    XAttack=np.array(dfAttack.drop(['classification'],1).astype(float))

    matriceDistTest=[]
    for i in range(len(X_test)):
        feature1=reshapeFeature(X_test[i])
        dist_matrixN = pairwise_distances_argmin_min(feature1,XNormal)
        dist_matrixA = pairwise_distances_argmin_min(feature1,XAttack)
        row=[X_test[i],XNormal[dist_matrixN[0].item()],XAttack[dist_matrixA[0].item()]]
        matriceDistTest.append(row)
    return matriceDistTest
    

tic=time.time() # recupera il tempo corrente in secondi

path = "D:\Tesi\dataset\DatasetAgg"

test = pd.read_csv(path+"\Test_encoded.csv", delimiter=",")
train = pd.read_csv(path+"\Train_encoded.csv", delimiter=",")


trainNormal = train[train['classification'] == 1]
trainAttack = train[train['classification'] == 0]

trainY = train['classification']
trainY=trainY.sort_values(ascending=False)
trainY = pd.DataFrame(trainY)
TrainY=trainY.values

testX, testY = getXY(test)

distTrain = findNearestTrain(train, trainNormal, trainAttack)
distTrain=np.array(distTrain)

distTest = findNearestTest(testX, trainNormal, trainAttack)
distTest=np.array(distTest)

saveNpArray(distTrain, trainY, "/TrainTr")
saveNpArray(distTest, testY, "/TestTr")

toc=time.time()
tempoTot = toc-tic
print(tempoTot)


