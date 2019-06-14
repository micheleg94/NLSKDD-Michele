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

CLUSTERS=0 #Se 1 le immagini vengono create tramite cluster, se 0 le immagini vengono create considerando gli esempi del training
TRAIN=1 #Se 1 vengono create immagini del training set
TEST=0 #Se 1 vengono create immagini del testing set
DISTANCE=0 #Se 0 non viene effettuate la parte di creazione di immagini
CNN=1 #se 0 non viene effettuato l'addrestramento ed il testing della CNN
SALVATAGGIO=1
LOAD_MODEL = 0
LOAD_CENTER = 0
num_cluster = (sys.argv[1]) #int(sys.argv[1])
print(num_cluster)
iteration=sys.argv[2]
#iteration=str(1)
nearest = True
path = "dataset\DatasetAgg"
pathModels = 'dataset\ModelCNN\\UNSW'

tempoFitCNN = 0
tempoTestCNN = 0

fileOutput=str(num_cluster)+'resultCNN.txt'
file = open(os.path.join(path+"\\CICIDS\\DS_"+iteration+"\\time\\", fileOutput), 'w')
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
        np.save(path + "\\CICIDS\\DS_"+iteration+"\\centerAtt"+str(num_cluster)+".npy", centerAtt)
        centerNorm=clusteringMiniBatchKMeans(X_Normal, num_cluster, row_normal)
        print("NormDone")
        np.save(path + "\\CICIDS\\DS_"+iteration+"\\centerNorm"+str(num_cluster)+".npy", centerNorm)
    elif LOAD_CENTER==1:
        centerAtt=np.load(path + "\\CICIDS\\DS_"+iteration+"\\centerAtt"+str(num_cluster)+".npy")
        centerNorm=np.load(path + "\\CICIDS\\DS_"+iteration+"\\centerNorm"+str(num_cluster)+".npy")
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
    train = pd.read_csv(path+"\\CICIDS\\DS_"+iteration+"\\Train_encoded.csv", delimiter=",")
    trainNormal = train[train['classification'] == 1]
    trainAttack = train[train['classification'] == 0]
    
    trainX, trainY = getXY(train)
    
    #for i in range(1,10):
    test = pd.read_csv(path+"\\CICIDS\\DS_"+iteration+"\\Test_encoded"+str(1)+".csv", delimiter=",")
    test.fillna((0), inplace=True)
            
            #divido il train in esempi normali e di attacco, utili per il K-Means
            
    testX, testY = getXY(test)
            
    distTrain, distTest = distance(trainX, testX, trainNormal, trainAttack,nearest)
    distTrain=np.array(distTrain)
    distTest=np.array(distTest)
        #for i in range(0,9):
         #   plt.subplot(330+1+i)
          #  plt.imshow(toimage(distTest[i]))
            
    if TRAIN==1:
        saveNpArray(distTrain, trainY, "/CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"Train")
        
    TEST=1
    TRAIN=0
    CLUSTERS=0
    LOAD_CENTER=0
    
    for i in range(1,10):
        test = pd.read_csv(path+"\\CICIDS\\DS_"+iteration+"\\Test_encoded"+str(i)+".csv", delimiter=",")
        test.fillna((0), inplace=True)
            
        testX, testY = getXY(test)

        distTrain, distTest = distance(trainX, testX, trainNormal, trainAttack,nearest)
        distTrain=np.array(distTrain)
        distTest=np.array(distTest)
        
        if TEST==1:    
            saveNpArray(distTest, testY, "/CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"Test"+str(i))


if CNN==1:
    tic=time.time()
    np.set_printoptions(precision=6)
    
    x_train = np.load(path+"//CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"TrainX.npy")
    y_train = np.load(path+"//CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"TrainY.npy")
    for i in range(1,10):
        x_test = np.load(path+"//CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"Test"+str(i)+"X.npy")
        y_test = np.load(path+"//CICIDS//DS_"+iteration+"//Cluster//"+str(num_cluster)+"Test"+str(i)+"Y.npy")
        
        
        
        
        batch_size = 32
        num_classes = 2
        epochs = 80
        img_rows, img_cols = 3, 10
        
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
        
        if LOAD_MODEL==0:
        
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True),
                # reduce_lr
            ]
            
            
    
            
            model = load_model('dataset\ModelCNN\modelNoPool.h5')
            #model = load_model('dataset\ModelCNN\modelPooling1.h5')
                    
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
            
            modelName = str(num_cluster) + 'cnn.h5'
            model.save(path+"/CICIDS//DS_"+iteration+"//"+ modelName)
            
            toc=time.time()
            file.write("Time Fitting CNN : "+str(toc-tic))
            file.write('\n')
        
        else:
            modelName = str(num_cluster) + 'cnn.h5'
            model=load_model(path+"/CICIDS//DS_"+iteration+"//"+ modelName)
            model.summary()
            
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
            val = (str(num_cluster), "PoolCICIDS-"+iteration, str(score[1]), str(tpr), str(fpr))
            mycursor.execute(sql, val)
                        
            mydb.commit()
        LOAD_MODEL=1


file.close()














