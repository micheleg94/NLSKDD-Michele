import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

np.random.seed(12)
from tensorflow import set_random_seed

set_random_seed(12)

from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler, Normalizer, normalize
from keras import optimizers

import scipy.stats as ss
from keras import callbacks
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Model
from keras.layers import Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras import optimizers
from keras.models import Model, load_model
from keras import regularizers


import seaborn as sns

LOAD_AUTOENCODER=0



# A deep autoecndoer model
def autoEncoder(x_train, params):
    n_col = x_train.shape[1]
    input = Input(shape=(n_col,))
    print(input)
    # encoder_layer
    # Dropoout?
    #  input1 = Dropout(.2)(input)
    encoded = Dense(params['first_layer'], activation=params['first_activation'],
                    kernel_initializer=params['kernel_initializer'],
                    name='encoder1')(input)
    encoded = Dense(params['second_layer'], activation=params['first_activation'],
                    kernel_initializer=params['kernel_initializer'],

                    name='encoder2')(encoded)
    encoded = Dense(params['third_layer'], activation=params['first_activation'],
                    kernel_initializer=params['kernel_initializer'],

                    name='encoder3')(encoded)
    
   # l1 = BatchNormalization()(encoded)
   # encoded = Dropout(.5)(encoded)
    decoded = Dense(params['second_layer'], activation=params['first_activation'], kernel_initializer=params['kernel_initializer'], name='decoder1')(encoded)
    decoded = Dense(params['first_layer'], activation=params['second_activation'], kernel_initializer=params['kernel_initializer'], name='decoder2')(decoded)
    decoded = Dense(n_col, activation=params['third_activation'], kernel_initializer=params['kernel_initializer'], name='decoder')(decoded)
    # serve per L2 normalization?
    # encoded1_bn = BatchNormalization()(encoded)

    autoencoder = Model(input=input, output=decoded)

    autoencoder.summary
    learning_rate = 0.001
    decay = learning_rate / params['epochs']
    autoencoder.compile(loss=params['losses'],
                        optimizer=params['optimizer']()#(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, amsgrad=False)#
                         , metrics=['accuracy'])

    return autoencoder


def getXY(train, test):
    clssList = train.columns.values
    target = [i for i in clssList if i.startswith('classification')]
    train_Y=train[target]
    # print(train_Y.head)
    test_Y=test[target]
    # remove label from dataset
    train_X = train.drop(target, axis=1)
    train_X=train_X.values
    #print(train_X.columns.values)
    test_X = test.drop(target, axis=1)
    test_X=test_X.values

    return train_X, train_Y, test_X, test_Y

def printPlotLoss(history, d):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("plotLoss" + str(d) + ".png")
    plt.close()
    # plt.show()


def printPlotAccuracy(history, d):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("plotAccuracy" + str(d) + ".png")
    plt.close()
    # plt.show()


def main():
    #split to validate model during training
    pathTrain='D:\Tesi\dataset\DatasetAgg\DatasetStand\Train_standard'
    pathTest = 'D:\Tesi\dataset\DatasetAgg\DatasetStand\Test_standard'
    train = pd.read_csv(pathTrain + ".csv")
    test = pd.read_csv(pathTest + ".csv")
    
    train_X, train_Y, test_X, test_Y= getXY(train, test)
        
    if LOAD_AUTOENCODER==0:
        VALIDATION_SPLIT = .2
    
        #pathTrain = 'KDDTrain+aggregateOneClsNumeric'

    
        callbacks_list = [
            # callbacks.ModelCheckpoint(
            #   filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            #  monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=4, restore_best_weights=True),
            # reduce_lr
        ]
    
    
    
        print('Model with autoencoder+softmax with training encoder weights')
        # parametri per autoencoder
        p = {
            'first_layer': 100,
            'second_layer': 40,
            'third_layer': 10,
            'batch_size': 64,
            'epochs': 150,
            'optimizer': optimizers.Adam,
            'kernel_initializer': 'glorot_uniform',
            'losses': 'mse',
            'first_activation': 'tanh',
            'second_activation': 'tanh',
            'third_activation': 'sigmoid'}
    
        autoencoder = autoEncoder(train_X, p)
        autoencoder.summary()
    
    
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder3').output)
        encoder.summary()    
    
        history = autoencoder.fit(train_X, train_X,
                                   validation_split=VALIDATION_SPLIT,
                                   batch_size=p['batch_size'],
                                   epochs=p['epochs'], shuffle=False,
                                   callbacks=callbacks_list,
                                   verbose=1,)
                                   #validation_data=(test_X, test_X))
                                   
        printPlotAccuracy(history, 'autoencoder')
        printPlotLoss(history, 'autoencoder')
        
        print(history)
        
        autoencoder.save('EncoderUN.h5')
    
    else:
        autoencoder = load_model('Encoder.h5')
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder3').output)
        encoder.summary()
    


    
    #estract encoder layers from autoEncoder    
    encoded_train = pd.DataFrame(encoder.predict(train_X))
    encoded_train = encoded_train.add_prefix('feature_')
    encoded_train["classification"]=train_Y
    print(encoded_train.shape)
    print(encoded_train.head())
    encoded_train.to_csv('train_encoded.csv', index=False)
    
    encoded_test = pd.DataFrame(encoder.predict(test_X))
    encoded_test = encoded_test.add_prefix('feature_')
    encoded_test["classification"] = test_Y
    print(encoded_test.shape)
    print(encoded_test.head())
    encoded_test.to_csv('test_encoded.csv', index=False)

    
    score = autoencoder.evaluate(test_X, test_X, verbose=1)
    print('Test loss normal:', score[0])
    print('Test accuracy normal:', score[1])
    
    scoreTrain = autoencoder.evaluate(train_X,train_X,verbose=1)
    print('Train loss normal:', scoreTrain[0])
    print('Train accuracy normal:', scoreTrain[1])
    




if __name__ == "__main__":
    main()