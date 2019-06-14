# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:07:24 2019

@author: miche
"""


from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
from clustering import clusteringKMeans, clusteringMiniBatchKMeans
import time
from matplotlib import pyplot as plt
from scipy.misc import toimage
from Utility import reshapeFeature, saveNpArray, loadNpArray,getXY
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from keras import backend as K
import keras
from keras.models import load_model
from keras import callbacks
import mysql.connector as msql
import os
import sys

np.random.seed(123)
from tensorflow import set_random_seed

set_random_seed(123)

mydb=msql.connect(host="localhost",
  user="root",
  passwd="toor")

mycursor = mydb.cursor()

mycursor.execute("USE risultati")

np.set_printoptions(precision=6)

CLUSTERS=1 #Se 1 le immagini vengono create tramite cluster, se 0 le immagini vengono create considerando gli esempi del training
TRAIN=1 #Se 1 vengono create immagini del training set
TEST=1 #Se 1 vengono create immagini del testing set
DISTANCE=1 #Se 0 non viene effettuate la parte di creazione di immagini
CNN=1 #se 0 non viene effettuato l'addrestramento ed il testing della CNN
SALVATAGGIO=1 
num_cluster = int(sys.argv[1])
print(num_cluster)
nearest = False
path = "dataset\DatasetAgg"

tempoFitCNN = 0
tempoTestCNN = 0

fileOutput=str(num_cluster)+'result.txt'
file = open(os.path.join(path+"\\time", fileOutput), 'w')
file.write('Result time for: %s clusters \n' %num_cluster)
file.write('\n')



def distance(X_Train, X_Test,dfNormal, dfAttack,nearest):
    #trasformo i dataframe in array
    #print (time.strftime("%H:%M:%S"))
    X_Normal=np.array(dfNormal.drop(['classification'],1).astype(float))
    X_Attack=np.array(dfAttack.drop(['classification'],1).astype(float))
    
    row_attack=np.size(X_Attack,0)
    row_normal=np.size(X_Normal,0)
    

    tic=time.time() # recupera il tempo corrente in secondi
    if CLUSTERS==1:
        centerAtt=clusteringMiniBatchKMeans(X_Attack, num_cluster, row_attack)
        print("AttDone")
        #print (time.strftime("%H:%M:%S"))
        centerNorm=clusteringMiniBatchKMeans(X_Normal, num_cluster, row_normal)
        print("NormDone")
        #print (time.strftime("%H:%M:%S"))
    else:
        centerAtt=X_Attack
        centerNorm=X_Normal
        
    toc=time.time()
    file.write("Time Creation Clusters: "+str(toc-tic))
    file.write('\n')

    matriceDistTrain=[]
    matriceDistTest=[]

#ciclo sull'intero dataset per calcolare per ogni esempio, i centroide più vicini
    tic=time.time()
    if TRAIN==1:
        for i in range(len(X_Train)):
            feature1=reshapeFeature(X_Train[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
            if nearest==True:
                if dist_matrixN[1]==0:
                    ind = dist_matrixN[0]
                    centerNorm[ind,:] = 0
                    dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
                    centerNorm[ind] = feature1
                
            dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
            if nearest==True:
                if dist_matrixA[1]==0:
                    ind = dist_matrixA[0]
                    centerAtt[ind,:] = 0
                    dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
                    centerAtt[ind] = feature1
            
            row=[X_Train[i],centerNorm[dist_matrixN[0].item()],centerAtt[dist_matrixA[0].item()]]
            matriceDistTrain.append(row)
        toc=time.time()
        file.write("Time Creation Training Images : "+str(toc-tic))
        file.write('\n')
    #print (time.strftime("%H:%M:%S"))
        
    tic=time.time()
    if TEST==1:
        for i in range(len(X_Test)):
            feature1=reshapeFeature(X_Test[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1,centerNorm)
            dist_matrixA = pairwise_distances_argmin_min(feature1,centerAtt)
            row=[X_Test[i],centerNorm[dist_matrixN[0].item()],centerAtt[dist_matrixA[0].item()]]
            matriceDistTest.append(row)
        toc=time.time()
        file.write("Time Creation Testing Images : "+str(toc-tic))
        file.write('\n')

    return matriceDistTrain, matriceDistTest
    



if DISTANCE==1:
    train = pd.read_csv(path+"\Train_encoded.csv", delimiter=",")
    test = pd.read_csv(path+"\Test_encoded.csv", delimiter=",")
    
    #divido il train in esempi normali e di attacco, utili per il K-Means
    trainNormal = train[train['classification'] == 1]
    trainAttack = train[train['classification'] == 0]
    
    trainX, trainY = getXY(train)
    testX, testY = getXY(test)
    
    distTrain, distTest = distance(trainX, testX, trainNormal, trainAttack,nearest)
    distTrain=np.array(distTrain)
    distTest=np.array(distTest)
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow(toimage(distTest[i]))
    if TRAIN==1:
        saveNpArray(distTrain, trainY, "/DatasetCl/"+str(num_cluster)+"Train")
    if TEST==1:    
        saveNpArray(distTest, testY, "/DatasetCl/"+str(num_cluster)+"Test")


if CNN==1:
    tic=time.time()
    np.set_printoptions(precision=6)
    
    x_train = np.load(path+"\DatasetCl\\"+str(num_cluster)+"TrainX.npy")
    y_train = np.load(path+"\DatasetCl\\"+str(num_cluster)+"TrainY.npy")
    x_test=np.load(path+"\DatasetCl\\"+str(num_cluster)+"TestX.npy")
    y_test=np.load(path+"\DatasetCl\\"+str(num_cluster)+"TestY.npy")
    
    
    
    
    batch_size = 32
    num_classes = 2
    epochs = 50
    img_rows, img_cols = 3, 10
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True),
        # reduce_lr
    ]
    
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
        
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test1 = keras.utils.to_categorical(y_test, num_classes)
    
    model = load_model('dataset\ModelCNN\modelNoPool.h5')
    #model = load_model('dataset\ModelCNN\modelPooling.h5')
            
    model.summary()
    
    model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks_list,
              verbose=2,
              #validation_data=(x_test, y_test1), 
              validation_split=0.1
              )
    toc=time.time()
    file.write("Time Fitting CNN : "+str(toc-tic))
    file.write('\n')
        
    tic=time.time()
    score = model.evaluate(x_test, y_test1, verbose=0)
        
    Y_test = model.predict_classes(x_test, verbose=1)
    
    matrix = confusion_matrix(y_test, Y_test)
    print("Accuracy is %f." % accuracy_score(y_test, Y_test))
    print(confusion_matrix(y_test, Y_test))
    tpr = ((matrix[0][0]) / (matrix[0][0] + matrix[0][1]))  # tpr 0.30
    print("TPR is", tpr)
    fpr=1 - recall_score(y_test, Y_test)
    print("FPR is ", fpr)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    toc=time.time()
    file.write("Time Testing CNN : "+str(toc-tic))
    file.write('\n')
    
    if SALVATAGGIO==1:
        sql = "INSERT INTO result VALUES (%s, %s, %s, %s, %s)"
        val = (str(num_cluster), "EqualSeed", str(score[1]), str(tpr), str(fpr))
        mycursor.execute(sql, val)
                    
        mydb.commit()


file.close()














