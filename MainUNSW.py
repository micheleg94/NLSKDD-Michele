# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:59:17 2019

@author: miche
"""

import pandas as pd
import numpy as np
from numpy import argmax
from PreProcessing import ohe, standardScale, minMaxScale


def main():
    
    path="dataset\DatasetAgg"
    
    
    train_df=pd.read_csv(path+"\\UNSW_NB15_training-set.csv",delimiter=",")
    test_df=pd.read_csv(path+"\\UNSW_NB15_testing-set.csv",delimiter=",")
    
    train_df = train_df.drop(['id','label'],axis=1)
    test_df = test_df.drop(['id','label'],axis=1)
    train_df = train_df.rename(columns={'attack_cat':'classification'})
    test_df = test_df.rename(columns={'attack_cat':'classification'})
    col_names=train_df.columns
    print(col_names)
    
    
    nominal_inx = [1, 2, 3]
    binary_inx = [36,41]
    numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))
    
    
    nominal_cols = col_names[nominal_inx].tolist()

    numeric_cols = col_names[numeric_inx].tolist()

    
    train_df, test_df = ohe(train_df, test_df, nominal_cols)
    train_df, test_df = standardScale(train_df, test_df, numeric_cols)
    
    train_df["classification"].replace(to_replace=dict(Normal=1, Reconnaissance=0, DoS=0, Exploits=0, Fuzzers=0, Shellcode=0, Analysis=0, Backdoor=0, Generic=0, Worms=0), inplace=True)
    test_df["classification"].replace(to_replace=dict(Normal=1, Reconnaissance=0, DoS=0, Exploits=0, Fuzzers=0, Shellcode=0, Analysis=0, Backdoor=0, Generic=0, Worms=0), inplace=True)
    
    train_df.to_csv(path+'\DatasetStand\Train_standard.csv', index=False)
    
    
    test_df.to_csv(path+'\DatasetStand\Test_standard.csv', index=False)
    
if __name__ == "__main__":
    main()





