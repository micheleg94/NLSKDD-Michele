# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:59:17 2019

@author: miche
"""

import pandas as pd
import numpy as np
from numpy import argmax
from PreProcessing import ohe, standardScale, minMaxScale, trainScaler, testScaler


def main():
    
    path="D:\Tesi\dataset\DatasetAgg\CICIDS"
    
    train_df=pd.read_csv(path+"\\DS_1\\train_CICIDS2017OneCls.csv",delimiter=",")
    test_df=pd.read_csv(path+"\\DS_1\\test1_CICIDS2017OneCls.csv",delimiter=",")
    
    train_df = train_df.rename(columns={' Flow Packets/s':'Flow_Packets', 'Flow Bytes/s':'Flow_Bytes',' Label':'Classification'})
    test_df = test_df.rename(columns={' Flow Packets/s':'Flow_Packets', 'Flow Bytes/s':'Flow_Bytes',' Label':'Classification'})
    train_df['Flow_Bytes'].fillna((0), inplace=True)
    train_df['Flow_Bytes'] = train_df['Flow_Bytes'].astype(float)
    
    Pack=train_df[train_df.Flow_Packets != 'Infinity']
    Bytes=train_df[train_df.Flow_Bytes != np.inf]

    maxPack = np.max(Pack['Flow_Packets'])
    maxBytes = np.max(Bytes['Flow_Bytes'])
    
    print(train_df.info())
    print(test_df.info())
        
    col_names=train_df.columns
    train_df['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack),inplace=True)
    train_df['Flow_Bytes'].replace((np.inf, maxBytes) ,inplace=True)
    #train_df['Flow_Bytes'].replace(to_replace=dict(np.inf=maxBytes),inplace=True)
    train_df["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)
    
    nominal_inx = []
    binary_inx = [30,31,32,33,43,44,45,46,47,48,49,50,56,57,58,59,60,61]
    numeric_inx = list(set(range(78)).difference(nominal_inx).difference(binary_inx))
        
    numeric_cols = col_names[numeric_inx].tolist()

    
    test_df['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack),inplace=True)
           
    test_df['Flow_Bytes'].replace(to_replace=dict(Infinity=maxBytes),inplace=True)
    test_df["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)
    
    
        
            
            #train_df, test_df = ohe(train_df, test_df, nominal_cols)
    train_df, test_df = standardScale(train_df, test_df, numeric_cols)
            
    #test_df = testScaler(test_df, numeric_cols, scaler)
            
            
            
    test_df.to_csv(path+'\\DS_11\\Test_standard.csv', index=False)
    train_df.to_csv(path+'\\DS_11\\Train_standard.csv', index=False)
    
    
if __name__ == "__main__":
    main()





