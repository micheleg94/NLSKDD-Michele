# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:40:34 2019

@author: miche
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import pandas as pd


def ohe(df, categories):
    for cat in categories:
        
        labEnc = LabelEncoder()
        df[cat+'_encoded'] = labEnc.fit_transform(df[cat])
        
        ohenc = OneHotEncoder()
        X = ohenc.fit_transform(df[cat+'_encoded'].values.reshape(-1,1)).toarray()
        X = X.astype(int)
        
        val=df[cat].values.tolist()

        val1=set(val)

        
        dfOneHot = pd.DataFrame(X, columns = [cat+"_"+i for i in sorted(val1)])
        if cat=="service" and len(val1)!=70:
            service_val=["service_aol", "service_harvest","service_http_2784","service_http_8001","service_red_i","service_urh_i"]
            dfSer = pd.DataFrame(0, index=range(22544),columns=service_val)
            dfOneHot = pd.concat([dfOneHot,dfSer], axis=1)
            dfOneHot = dfOneHot.reindex(sorted(dfOneHot.columns), axis=1)
            
            
        df=df.drop(cat+'_encoded', axis=1)
        df = pd.concat([df, dfOneHot], axis=1)
        df= df.drop(cat, axis=1)
        
    return df


def standardScale(df, categories):
    scaler = StandardScaler()
    df[categories] = scaler.fit_transform(df[categories])  
    return df


def minMaxScale(df, categories):
    scaler = MinMaxScaler()
    
    df[categories] = scaler.fit_transform(df[categories])
    
    return df
    