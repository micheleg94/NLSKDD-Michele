# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:26:47 2019

@author: miche
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


num_classes = 2
img_rows, img_cols = 3, 10


input_shape = (img_rows, img_cols, 1)

#Creazione primo modello senza pooling

model = Sequential()

model.add(Conv2D(32, kernel_size=(2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Dropout(0.3)) 
model.add(Conv2D(16, kernel_size=(2,4), activation='relu')),
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.save('modelNoPool.h5')
model.summary()

#creazione secondo modello con pooling

model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(2),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Conv2D(64, kernel_size=(1, 3), activation='relu')),
model1.add(Flatten())
model1.add(Dropout(0.5))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(num_classes, activation='softmax'))

model1.save('modelPooling.h5')
model1.summary()











