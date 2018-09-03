# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:26:25 2018

@author: Xiaoyun Li
"""

import numpy as np
import pandas as pd
import math
import rpy2
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def purity_score(x,y_true):
    n_class = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_class, random_state=None).fit(x)
    y_pred = kmeans.labels_
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix), \
           normalized_mutual_info_score(y_true.flatten(), y_pred.flatten())

def compute_dist(X):        #rows: samples   columns: features
    D = pairwise_distances(X)
    D /= np.max(D)
    return D
#    return 1-rbf_kernel(X, gamma=1)

def pair_dist(x1,x2,Mm):      #used for entropy
    d=sum((x1-x2)**2/Mm**2) 
    return d

def entropy(X):
    n,d=X.shape
    Mm=np.max(X,0)-np.min(X,0)
    E=0
    D=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i,j]=pair_dist(X[i,:],X[j,:],Mm)
    
    alpha = -np.log(0.5)/np.mean(D)
    sim = np.exp(-alpha*D)
    for i in range(n):
        for j in range(n):
            if i!=j:
                E = E-(sim[i,j]*np.log(sim[i,j])+(1-sim[i,j])*np.log(1-sim[i,j]))/n/(n-1)
    return E

def class_seperability(X,y):
    n,d=X.shape
    classes = np.unique(y)
    num_class = len(classes)
    prior = np.zeros([num_class,1])
    mu = np.zeros([num_class,d])    #class center
    M = np.mean(X,0)     #overall center
    Sw_c = []
    Sb_c = []
    for i in range(num_class):
        c = classes[i]
        prior[i] = np.sum(y == c)/n
        class_X = np.array([a for k,a in enumerate(X) if y[k]==c])
        mu[i,:] = np.mean(class_X,0)
        temp = np.zeros([d,d])
        for k in range(class_X.shape[0]):
            x = class_X[k,:]
            temp = temp+np.matmul(np.expand_dims(x-mu[i,:],0).T,np.expand_dims(x-mu[i,:],0))
            
        Sw_c.append(temp)
        Sb_c.append(np.matmul(np.expand_dims(mu[i,:]-M,0).T,np.expand_dims(mu[i,:]-M,0)))
    Sw = sum([a*b for a,b in zip(prior,Sw_c)])
    Sb = sum(Sb_c)
    S = np.matmul(np.linalg.inv(Sw+np.diag([1e-8]*Sw.shape[0])),Sb)
    return np.trace(S)
    
def knn_accuracy(x_test,y_test,select_feature,k=3):
    n,d=x_test.shape
    neigh = KNeighborsClassifier(n_neighbors=k)
    y_test = y_test.reshape(-1,)
    neigh.fit(x_test[:,select_feature],y_test)
    fitted_y = neigh.predict(x_test[:,select_feature])
    return np.sum(np.equal(fitted_y, y_test))/n

def connectivity(X_old,X_new,pctg=0.1,two_sided=True):
    
    dist_matrix = compute_dist(X_old)   
    dist_matrix_new = compute_dist(X_new)
    if two_sided:

        threshold_high = np.percentile(dist_matrix,100-pctg*100)
        threshold_low = np.percentile(dist_matrix,pctg*100)
        np.fill_diagonal(dist_matrix,np.percentile(dist_matrix,50))
        original_connection = (dist_matrix > threshold_high) | (dist_matrix < threshold_low)    
    
        threshold_high = np.percentile(dist_matrix_new,100-pctg*100)
        threshold_low = np.percentile(dist_matrix_new,pctg*100)
        np.fill_diagonal(dist_matrix_new,np.percentile(dist_matrix_new,50))
        new_connection = (dist_matrix_new > threshold_high) | (dist_matrix_new < threshold_low)  
    else:
        threshold = np.percentile(dist_matrix,100-pctg*100)
        np.fill_diagonal(dist_matrix,0)
        original_connection = (dist_matrix > threshold)
    
        threshold = np.percentile(dist_matrix_new,100-pctg*100)
        np.fill_diagonal(dist_matrix_new,0)
        new_connection = (dist_matrix_new > threshold)
        
    still_connected =np.array(original_connection & new_connection,dtype=int)
    
    return(np.sum(still_connected)/np.sum(original_connection))

def compute_dgm(X, t = 0.2, max_dimension = 1, threshold=0.05):
    rpy2.robjects.numpy2ri.activate()
    dist_matrix = compute_dist(X)
    TDA = importr('TDA')
    dgm_list = TDA.ripsDiag(dist_matrix ,max_dimension,t,dist = "arbitrary",library='GUDHI')
    dgm = np.array(dgm_list.rx2('diagram'))
    dgm = np.array([a for a in dgm if a[2]-a[1]>threshold])
    return dgm

def dgm_distance(dgm1, dgm2, distance='W', dimension = 1, l=1):
    rpy2.robjects.numpy2ri.activate()
    TDA = importr('TDA')
#    dgm1=numpy2ri(dgm1)
#    dgm2=numpy2ri(dgm2)
    if distance=='W':
        d = np.array(TDA.wasserstein(dgm1, dgm2, p = l, dimension = dimension))
    else:
        d = np.array(TDA.bottleneck(dgm1, dgm2, dimension = dimension))
    return d

def dif_dist(D1,D2,norm='l1'):
    if norm == 'l1':
        return np.sum(np.abs(D1-D2))
    elif norm =='l2':
        return np.sum((D1-D2)**2)
    else:
        return np.max(np.abs(D1-D2))

