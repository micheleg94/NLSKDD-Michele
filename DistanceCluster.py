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
import time
from matplotlib import pyplot as plt
from scipy.misc import toimage
from Utility import reshapeFeature, saveNpArray, loadNpArray,getXY




def distance(X_Train, X_Test,dfNormal, dfAttack):
    #trasformo i dataframe in array
    X_Normal=np.array(dfNormal.drop(['classification'],1).astype(float))
    X_Attack=np.array(dfAttack.drop(['classification'],1).astype(float))
    
    centerAtt=clusteringKMeans(X_Attack, 1)
    print("AttDone")
    centerNorm=clusteringKMeans(X_Normal, 1)
    print("NormDone")

    matriceDistTrain=[]
    matriceDistTest=[]

#ciclo sull'intero dataset per calcolare per ogni esempio, i centroide più vicini
    for i in range(len(X_Train)):
        feature1=reshapeFeature(X_Train[i])
        dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
        dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
        row=[X_Train[i],centerNorm[dist_matrixN[0].item()],centerAtt[dist_matrixA[0].item()]]
        matriceDistTrain.append(row)
        
        
    for i in range(len(X_Test)):
        feature1=reshapeFeature(X_Test[i])
        dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
        dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
        row=[X_Test[i],centerNorm[dist_matrixN[0].item()],centerAtt[dist_matrixA[0].item()]]
        matriceDistTest.append(row)

    return matriceDistTrain, matriceDistTest



tic=time.time() # recupera il tempo corrente in secondi

path = "D:\Tesi\dataset\DatasetAgg"
train = pd.read_csv(path+"\Train_encoded.csv", delimiter=",")
test = pd.read_csv(path+"\Test_encoded.csv", delimiter=",")

#divido il train in esempi normali e di attacco, utili per il K-Means
trainNormal = train[train['classification'] == 1]
trainAttack = train[train['classification'] == 0]


trainX, trainY = getXY(train)
testX, testY = getXY(test)

distTrain, distTest = distance(trainX, testX, trainNormal, trainAttack)
distTrain=np.array(distTrain)
distTest=np.array(distTest)
for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(toimage(distTest[i]))

saveNpArray(distTrain, trainY, "/DatasetCluster/C10000/Train")
saveNpArray(distTest, testY, "/DatasetCluster/C10000/Test")


toc=time.time()
tempoTot = toc-tic
print(tempoTot)








