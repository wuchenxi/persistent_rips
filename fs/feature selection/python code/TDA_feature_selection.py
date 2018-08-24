# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:35:48 2018

@author: Xiaoyun Li
"""

from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.sparse_learning_based import UDFS
from sklearn.metrics.pairwise import rbf_kernel
import scipy.io
import dionysus as dy
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import time
import math

import numpy as np
import pandas as pd
import math
import evaluation_functions as ef
from CUT_SPEC import cut_spec
import sys
import warnings

#np.random.seed(168004147)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def standardize_feature(X):
    m = np.mean(X,0)
    sd = np.std(X,0)
    return (X-m)/sd
    
def compute_dist(X):        #rows: samples   columns: features
#    n,d=X.shape
#    A=np.matmul(X,X.T)
#    d=np.diag(A)
#    B=np.tile(d,(n,1)).T
#    return np.sqrt(B+B.T-2*A)
    return 1-rbf_kernel(X, gamma=1)

def simulate_ball(n,d,d_noise,r=5,sigma=1):
    X=np.zeros([n,d+d_noise])
    for k in range(n):
        x=list(range(d+d_noise))
        remain=r**2
        for i in range(d):
            x1=np.random.uniform(-np.sqrt(remain),np.sqrt(remain),1)
            remain=r**2-x1**2
            x[i]=x1
        x[-d_noise:]=np.random.normal(0,sigma,d_noise)
        X[k,:]=x
    return X

def backward_distance_selection(x, target_dim, pctg=0.1, pack_size=1):    #X row:sample   col:feature
    n,d=x.shape  
    throw_list = []
    names = list(range(x.shape[1]))  
    feature_set = names
    final_score=[]
    for k in range(int((d-target_dim)/pack_size)):
        if k%5==0:
            print('iteration '+str(k+1))
        dist_matrix = compute_dist(x)
#        np.fill_diagonal(dist_matrix,math.inf)
        threshold = np.percentile(dist_matrix,100-pctg*100)
        original_connection = (dist_matrix > threshold)
        
        score=[]
        for feature in feature_set:
            ind = [i for i in range(len(feature_set)) if feature_set[i] !=feature]
            x_reduced = x[:,ind]
            reduced_dist=compute_dist(x_reduced)
#            np.fill_diagonal(dist_matrix,math.inf)
            threshold = np.percentile(reduced_dist,100-pctg*100)
            reduced_connection = (reduced_dist>threshold)
            still_connected =np.array(original_connection & reduced_connection,dtype=int)
            sample_persistence = [sum(a) for a in still_connected]
            score.append(sum(sample_persistence))
        
        score = np.array(score)
        idx = np.argsort(score)
        throw_feature = [feature_set[i] for i in range(len(feature_set)) if i in idx[-pack_size:]]
        throw_list = throw_list+throw_feature
        ind = [i for i in range(len(feature_set)) if feature_set[i] not in throw_feature]
        feature_set = [a for a in feature_set if a not in throw_feature]
            
#        throw_feature = feature_set[score.index(max(score))]
#        throw_list.append(throw_feature)
#        ind = [i for i in range(len(feature_set)) if feature_set[i] != throw_feature]
#        feature_set = [a for a in feature_set if a != throw_feature]
        
        x = x[:,ind]
        score = [a for a in score if a!=np.max(score)]
        final_score = score
        
    selected_list = [a for a in names if a not in throw_list] 
    selected_list = [a for _,a in sorted(zip(final_score,selected_list))]
    return throw_list , selected_list

def random_selection(x, target_dim, N, num_use, pctg=0.1, two_sided=True):
    n,d=x.shape  
    
    dist_matrix = compute_dist(x)
    if two_sided:
        threshold_high = np.percentile(dist_matrix,100-pctg*100)
        threshold_low = np.percentile(dist_matrix,pctg*100)
        np.fill_diagonal(dist_matrix,np.percentile(dist_matrix,50))
        original_connection = (dist_matrix > threshold_high) | (dist_matrix < threshold_low)
    else:
        threshold = np.percentile(dist_matrix,100-pctg*100)
        np.fill_diagonal(dist_matrix,0)
        original_connection = (dist_matrix > threshold)
    
    feature_score = np.ones(d)
    feature_counter = np.zeros(d)
    for k in range(N):
        use_feature = list(np.random.choice(list(range(d)),num_use,replace=False))
        feature_counter[use_feature]=feature_counter[use_feature]+1
        x_reduced = x[:,use_feature]
        reduced_dist=compute_dist(x_reduced)
        if two_sided:
            threshold_high = np.percentile(reduced_dist,100-pctg*100)
            threshold_low = np.percentile(reduced_dist,pctg*100)
            np.fill_diagonal(dist_matrix,np.percentile(dist_matrix,50))
            reduced_connection = (reduced_dist>threshold_high) | (reduced_dist < threshold_low)
        else:
            threshold = np.percentile(reduced_dist,100-pctg*100)
            np.fill_diagonal(reduced_dist,0)
            reduced_connection = (reduced_dist>threshold)
            
        still_connected =np.array(original_connection & reduced_connection,dtype=int)
        sample_persistence = [sum(a) for a in still_connected]
        score=sum(sample_persistence)      
        feature_score[use_feature]=feature_score[use_feature]+score/N
    
    feature_score = feature_score/feature_counter
    selected_list = np.argsort(feature_score)[-target_dim:]
    return selected_list

def compare_methods(x,y,num_select,pctg=0.1,pack_size=1,num_clusters=5,two_sided=False):
    
    n,d = x.shape
    idx = np.random.permutation(n)
    x,y = x[idx], y[idx]
    
    #########  split train and test  #########
    X=x;Y=y
    train_num = int(n*0.7)
    test_num = n-int(n*0.7)
    x=X[:train_num,:]; y=Y[:train_num]
    x_test = X[-test_num:,:];y_test = Y[-test_num:]
    
    ###########  other methods  ######################
    '''    Similarity based: lap_score  SPEC          '''
    start_time = time.clock()
    lap_score_result = lap_score.lap_score(x)
    lap_score_result= np.argsort(lap_score_result)[:num_select]
    print('lap_score running time:',time.clock()-start_time)
    
#    _,stepwise = backward_distance_selection(x,num_select,pctg,pack_size)   #pctg controls sensitivity to outliers
    
    start_time = time.clock()
    rf_result = random_selection(x, num_select, N=300, num_use=int(d/2),pctg=pctg, two_sided=two_sided)
    print('rf running time:',time.clock()-start_time)
    
    start_time = time.clock()
    SPEC_result = SPEC.spec(x)
    print('SPEC running time:',time.clock()-start_time)
    SPEC_result= np.argsort(SPEC_result)[:num_select]     #find minimum

    start_time = time.clock()
    CSPEC_result = cut_spec(x,pctg=0.15)
    print('cut-SPEC running time:',time.clock()-start_time)
    CSPEC_result= np.argsort(CSPEC_result)[:num_select]     #find minimum
    
    '''sparse learning based'''
    start_time = time.clock()
    MCFS_W = MCFS.mcfs(x,num_select)
    print('MCFS running time:',time.clock()-start_time)
    MCFS_result = [np.max(np.abs(x)) for x in MCFS_W]     #find maximum
    MCFS_result= np.argsort(MCFS_result)[-num_select:]

#    start_time = time.clock()
#    NDFS_W = NDFS.ndfs(x,**{'n_clusters':num_clusters})
#    print('NDFS running time:',time.clock()-start_time)
#    NDFS_result = [np.sqrt(np.sum(x**2)) for x in NDFS_W]     #find maximum
#    NDFS_result= np.argsort(NDFS_result)[-num_select:]
#
#    start_time = time.clock()
#    UDFS_W = UDFS.udfs(x,**{'n_clusters':num_clusters}) 
#    print('UDFS running time:',time.clock()-start_time)             
#    UDFS_result = [np.sqrt(np.sum(x**2)) for x in UDFS_W]     #find minimum ??????????????????????
#    UDFS_result= np.argsort(UDFS_result)[:num_select]
    
#    prop_x = x[:,list(stepwise)]
    rf_x = x[:,list(rf_result)]
    lap_score_x = x[:,list(lap_score_result)]
    SPEC_x = x[:,list(SPEC_result)]
    CSPEC_x = x[:,list(CSPEC_result)]
    MCFS_x = x[:,list(MCFS_result)]
#    NDFS_x = x[:,list(NDFS_result)]
#    UDFS_x = x[:,list(UDFS_result)]
    
    print('\n')
    print('Class Seperability')
#    print('prop', ef.class_seperability(prop_x,y))
    print('rf', ef.class_seperability(rf_x,y))
    print('lap_score', ef.class_seperability(lap_score_x,y))
    print('SPEC', ef.class_seperability(SPEC_x,y))
    print('cut-SPEC', ef.class_seperability(CSPEC_x,y))
    print('MCFS',ef.class_seperability(MCFS_x,y))
#    print('NDFS',ef.class_seperability(NDFS_x,y))
#    print('UDFS',ef.class_seperability(UDFS_x,y))  
    
    print('\n')
    print('KNN accuracy')
#    print('prop', ef.knn_accuracy(prop_x,y))
    print('rf', ef.knn_accuracy(x_test,y_test,rf_result))
    print('lap_score', ef.knn_accuracy(x_test,y_test,lap_score_result))
    print('SPEC', ef.knn_accuracy(x_test,y_test,SPEC_result))
    print('cut-SPEC', ef.knn_accuracy(x_test,y_test,CSPEC_result))
    print('MCFS',ef.knn_accuracy(x_test,y_test,MCFS_result))
#    print('NDFS',ef.knn_accuracy(x_test,y_test,NDFS_result))
#    print('UDFS',ef.knn_accuracy(x_test,y_test,UDFS_result),'\n')  

    print('\n')
    print('connectivity')
#    print('prop', ef.knn_accuracy(prop_x,y))
    print('rf', ef.connectivity(x,rf_x,pctg, two_sided))
    print('lap_score', ef.connectivity(x,lap_score_x,pctg, two_sided))
    print('SPEC', ef.connectivity(x,SPEC_x,pctg, two_sided))
    print('cut-SPEC', ef.connectivity(x,CSPEC_x,pctg, two_sided))
    print('MCFS',ef.connectivity(x,MCFS_x,pctg, two_sided))
#    print('NDFS',ef.connectivity(x,NDFS_x,pctg, two_sided))
#    print('UDFS',ef.connectivity(x,UDFS_x,pctg, two_sided),'\n')  
    
#    print('\n')
#    print('Entropy')
##    print('prop', ef.entropy(prop_x))
#    print('rf', ef.entropy(rf_x))
#    print('lap_score', ef.entropy(lap_score_x))
#    print('SPEC', ef.entropy(SPEC_x))
#    print('MCFS',ef.entropy(MCFS_x))
##    print('NDFS',ef.entropy(NDFS_x))
#    print('UDFS',ef.entropy(UDFS_x),'\n')  
    
    
#num_select=3
#ball = simulate_ball(200,3,5,r=5,sigma=1)
##ball[0,5]=1000
##ball = (ball - np.mean(ball, axis=0)) / np.std(ball, axis=0)
#compare_methods(ball,3)

#########  datasets  from package##############
datasets = os.listdir('D:\Anaconda3\Lib\site-packages\skfeature\data')
#datasets = ['ALLAML.mat']
shape = []
num_class = []
for dataset in datasets:
    data = scipy.io.loadmat('D:\Anaconda3\Lib\site-packages\skfeature\data\\'+dataset)
    X = data['X']
    y = data['Y']
    shape.append(X.shape)
    c = len(np.unique(y))
    num_class.append(c)
    if X.shape[0]<2000 and X.shape[1]<5000:
        n,d = X.shape
#        X = normalize(X,norm='l2',axis=0)
        X = standardize_feature(X)
        print(dataset)
        print(X.shape,len(np.unique(y)))
        if d>500:
            target_dim = 200
        elif d>100:
            target_dim = 50
        elif d>40:
            target_dim = 20
        elif d>20:
            target_dim = 10
        else:
            target_dim = 5
        compare_methods(X,y,num_select=target_dim,pctg=0.15,pack_size=1, num_clusters=c, two_sided=False)
    

#########  datasets from UCI  ##############
#print('+++++++++++++UCI data+++++++++++++++++++\n')
#datasets = os.listdir('..\data')
#data = []
#label = []
#for k,dataset in enumerate(datasets):
#    print(dataset)
#    data1 = pd.read_table('..\data\\'+dataset,sep=",", header= None, error_bad_lines=False)
#    data1 = data1.replace('?',np.nan).dropna()
#
#    if k in [0,1,2,7]:
#        data1 = data1.iloc[:,1:]
#        
#    if k in [3,4]:
#        data1 = pd.read_table('..\data\\'+dataset,sep=",", error_bad_lines=False)
#        
#    if dataset != 'wdbc.data':
#        y = data1.iloc[:,-1]
#        X = data1.iloc[:,:-1]
#    else:
#        y=data1.iloc[:,0]
#        X=data1.loc[:,data1.columns != 1]
#
#    if dataset =='drug_consumption.data':
#        y = data1.iloc[:,12]
#        data1 = data1.select_dtypes(include=['number'])
#        X = data1
#    
#    X = X.loc[:, (X != X.iloc[0]).any()] 
#    X = np.array(X,'float64')
#    y = np.array(y)
##    X = normalize(X,axis=0)
#    X = standardize_feature(X)
#
#    print(X.shape)
#    n,d = X.shape
#    
#    data.append(X)
#    label.append(y)
#    
#    num_class = len(np.unique(y))
#    if d>500:
#        target_dim = 200
#    elif d>100:
#        target_dim = 50
#    elif d>40:
#        target_dim = 20
#    elif d>20:
#        target_dim = 10
#    else:
#        target_dim = 5
#    compare_methods(X,y,num_select=target_dim,pctg=0.15,pack_size=1, num_clusters=num_class, two_sided=False)







