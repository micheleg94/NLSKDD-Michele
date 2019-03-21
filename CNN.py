# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:49:50 2019

@author: miche
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from keras import backend as K
import keras
import time
from keras.models import load_model

tic=time.time() # recupera il tempo corrente in secondi

path = "D:\Tesi\dataset\DatasetAgg"
x_train = np.load(path+"\Train\ClusterNo\TrainX.npy")
y_train = np.load(path+"\Train\ClusterNo\TrainY.npy")
x_test=np.load(path+"\Train\ClusterNo\TestX.npy")
y_test=np.load(path+"\Train\ClusterNo\TestY.npy")

batch_size = 32
num_classes = 2
epochs = 15
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

model = load_model('dataset\ModelCNN\modelNoPoll.h5')
#model = load_model('dataset\ModelCNN\modelPolling.h5')
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test1))

score = model.evaluate(x_test, y_test1, verbose=0)

Y_test = model.predict_classes(x_test, verbose=1)

matrix = confusion_matrix(y_test, Y_test)
print("Accuracy is %f." % accuracy_score(y_test, Y_test))
print(confusion_matrix(y_test, Y_test))
tpr = ((matrix[0][0]) / (matrix[0][0] + matrix[0][1]))  # tpr 0.30
print("TPR is", tpr)
print("FPR is ", 1 - recall_score(y_test, Y_test))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

toc=time.time()
tempoTot = toc-tic
print(tempoTot)










