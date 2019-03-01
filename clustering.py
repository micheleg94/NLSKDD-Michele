# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:37:15 2019

@author: miche
"""

from sklearn.cluster import KMeans 


def clusteringKMeans(X, i):
    kmeansAtt = KMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=12345).fit(X)
    
    return kmeansAtt.cluster_centers_




