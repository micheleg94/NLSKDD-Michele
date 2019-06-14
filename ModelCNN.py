# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:26:47 2019

@author: miche
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


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

#model.save('modelNoPool.h5')
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

#model1.save('modelPooling.h5')
model1.summary()

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(2),
                 activation='relu',
                 input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.5))
model2.add(Conv2D(32, kernel_size=(1, 3), activation='relu')),
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(30, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

#model2.save('modelPooling1.h5')
model2.summary()


model3 = Sequential()

model3.add(Conv2D(32, kernel_size=(2),
                 activation='relu',
                 input_shape=input_shape))
model3.add(Dropout(0.3)) 
model3.add(Conv2D(16, kernel_size=(2,4), activation='relu'))
model3.add(Dropout(0.3))
model3.add(Conv2D(8, kernel_size=(1,4), activation='relu'))
model3.add(Flatten())
model3.add(Dense(num_classes, activation='softmax'))

#model3.save('modelNoPool1.h5')
model3.summary()

model4 = Sequential()

model4.add(Conv2D(32, kernel_size=(2),
                 activation='relu',
                 input_shape=input_shape))
model4.add(AveragePooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))
model4.add(Conv2D(64, kernel_size=(1, 3), activation='relu')),
model4.add(Flatten())
model4.add(Dropout(0.5))
model4.add(Dense(20, activation='relu'))
model4.add(Dense(num_classes, activation='softmax'))

#model4.save('modelAvgPooling.h5')
model4.summary()











