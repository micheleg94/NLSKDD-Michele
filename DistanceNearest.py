# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:30:01 2019

@author: miche
"""


from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
import time
from DistanceCluster import saveNpArray



def reshapeFeature(x):
    feature=x.reshape(1,-1)
    
    return feature

def findNearest(df,dfNormal, dfAttack):
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

tic=time.time() # recupera il tempo corrente in secondi

test = pd.read_csv("D:\Tesi\dataset\DatasetAgg\Test_encoded.csv", delimiter=",")
train = pd.read_csv("D:\Tesi\dataset\DatasetAgg\Train_encoded.csv", delimiter=",")


trainNormal = train[train['classification'] == 1]
trainAttack = train[train['classification'] == 0]

testNormal = test[test['classification'] == 1]
testAttack = test[test['classification'] == 0]

trainY = train['classification']
trainY=trainY.sort_values(ascending=False)
trainY = pd.DataFrame(trainY)
TrainY=trainY.values

testY = test['classification']
testY = testY.sort_values(ascending=False)
testY = pd.DataFrame(testY)
testY = testY.values

distTrain = findNearest(train, trainNormal, trainAttack)
distTrain=np.array(distTrain)

distTest = findNearest(test, testNormal, testAttack)
distTest=np.array(distTest)

#saveNpArray(dist, Y)
saveNpArray(distTrain, trainY, "/Train")
saveNpArray(distTest, testY, "/Test")

toc=time.time()
tempoTot = toc-tic
print(tempoTot)


