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
    col_names =np.array(["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","classification."])
    train_df=pd.read_csv("D:\Tesi\dataset\DatasetAgg\KDDTrain+aggregate.csv",delimiter=",")
    
    
    test_df=pd.read_csv("D:\Tesi\dataset\DatasetAgg\KDDTest+aggregate.csv",delimiter=",")

    
    
    nominal_inx = [1, 2, 3]
    binary_inx = [6, 11, 13, 14, 20, 21]
    numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))
    
    
    nominal_cols = col_names[nominal_inx].tolist()
    #binary_cols = col_names[binary_inx].tolist()
    numeric_cols = col_names[numeric_inx].tolist()

    
    train_df = ohe(train_df, nominal_cols)
    train_df = standardScale(train_df, numeric_cols)
    #copyOfTrain = minMaxScale(copyOfTrain, numeric_cols)
    train_df["classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0), inplace=True)
    
    train_df.to_csv('D:\Tesi\dataset\DatasetAgg\DatasetStand\Train_standard.csv', index=False)
    
    test_df = ohe(test_df, nominal_cols)
    test_df = standardScale(test_df, numeric_cols)
    #copyOfTrain = minMaxScale(copyOfTrain, numeric_cols)
    test_df["classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0), inplace=True)
    
    test_df.to_csv('D:\Tesi\dataset\DatasetAgg\DatasetStand\Test_standard.csv', index=False)
    
if __name__ == "__main__":
    main()





