# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:26:25 2018

@author: Xiaoyun Li
"""

import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel

def compute_dist(X):        #rows: samples   columns: features
#    n,d=X.shape
#    A=np.matmul(X,X.T)
#    d=np.diag(A)
#    B=np.tile(d,(n,1)).T
#    return np.sqrt(B+B.T-2*A)
    return 1-rbf_kernel(X, gamma=1)

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

#def dgm_distance(X, old_diagram, keep_pctg=0.2):
    