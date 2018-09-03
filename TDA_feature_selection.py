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
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
import evaluation_functions as ef
from CUT_SPEC import cut_spec
import sys
import warnings
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from scipy.stats import kendalltau
#from scipy.metric import ndcg_score

np.random.seed(1993)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
##########  install R packages in Anaconda environment for rpy2  ###########
#utils = importr('utils')
#utils.install_packages('TDA')

#############################################
def create_folder(name):
    if not os.path.exists('./result/'+name):
        os.makedirs('./result/'+name)
    
def standardize_feature(X):
    m = np.mean(X,0)
    sd = np.std(X,0)
    return (X-m)/sd
    
def compute_dist(X):        #rows: samples   columns: features
    D = pairwise_distances(X)
    D /= np.max(D)
    return D
#    return 1-rbf_kernel(X, gamma=1)

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


def random_selection(x, target_dim, N, num_use, pctg=0.5, two_sided=True):
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
    selected_list = np.argsort(feature_score)[-target_dim:]    #find max
    return selected_list

def ranking_selection(x, target_dim, N, num_use, sample_pctg=0.5, preserve_pctg = 0.5, dist = 'l2'):
    n,d=x.shape
    dist_matrix = compute_dist(x)
    original_rank_all = np.array([a.argsort() for a in dist_matrix])
    original_rank_index = np.array([a[-int(n*preserve_pctg):] for a in original_rank_all])
    original_rank_full = np.array([a.argsort() for a in original_rank_index])
    original_dist_at_index = np.array([a[b] for (a,b) in zip(dist_matrix, original_rank_index)])
    feature_score = np.ones(d)   
    feature_score_l1 = np.ones(d)
    feature_score_l2 = np.ones(d)
    feature_score_lmax = np.ones(d)
    
    feature_counter = np.zeros(d)
    for k in range(N):
        use_feature = list(np.random.choice(list(range(d)),num_use,replace=False))
        feature_counter[use_feature]=feature_counter[use_feature]+1
        x_reduced = x[:,use_feature]
        reduced_dist=compute_dist(x_reduced)            
        eval_sample = list(np.random.choice(list(range(n)),int(sample_pctg*n),replace=False))
        original_rank = original_rank_full[eval_sample,:]
        original_dist = original_dist_at_index[eval_sample,:]
        reduced_rank = np.array([a.argsort() for a in reduced_dist[eval_sample,:]]) 
        reduced_rank_at_index = np.array([a[b] for (a,b) in zip(reduced_rank,original_rank_index[eval_sample,:])])
        reduced_rank = np.array([a.argsort() for a in reduced_rank_at_index])
        
        reduced_dist_at_index = np.array([a[b] for (a,b) in zip(reduced_dist[eval_sample,:], original_rank_index[eval_sample,:])])
        dif = original_dist- reduced_dist_at_index
        
        tau = [kendalltau(a,b)[0] for a,b in zip(original_rank,reduced_rank)]            
        score = sum(tau)
        score_l1 = np.sum(np.abs(dif))
        score_l2 = np.sqrt(np.sum(dif**2))
        score_lmax = np.max(np.abs(dif))
        
        feature_score[use_feature]=feature_score[use_feature]+score/N
        feature_score_l1[use_feature]=feature_score_l1[use_feature]+score_l1/N
        feature_score_l2[use_feature]=feature_score_l2[use_feature]+score_l2/N
        feature_score_lmax[use_feature]=feature_score_lmax[use_feature]+score_lmax/N
    
    feature_score /= feature_counter; feature_score_l1 /= feature_counter
    feature_score_l2 /= feature_counter; feature_score_lmax /= feature_counter
    
    selected_list = np.argsort(feature_score)[-target_dim:]   #find max
    selected_list_l1 = np.argsort(feature_score_l1)[:target_dim]   #find min
    selected_list_l2 = np.argsort(feature_score_l2)[:target_dim]   #find min
    selected_list_lmax = np.argsort(feature_score_lmax)[:target_dim]   #find min
    return selected_list, selected_list_l1, selected_list_l2, selected_list_lmax

def compare_methods(x,y,num_select,pctg=0.5,sample_pctg=1, num_clusters=5,zero_mean=False,dim=1,t=0.8,thresh=0.1):
    if zero_mean == False:
        x = normalize(x,axis=0)
    else:
        x = standardize_feature(x)
        
    n,d = x.shape
    
#    idx = np.random.permutation(n)
#    x,y = x[idx], y[idx]
#    
#    #########  split train and test  #########
#    X=x;Y=y
#    train_num = int(n*0.6)
#    test_num = n-int(n*0.6)
#    x=X[:train_num,:]; y=Y[:train_num]
#    x_test = X[-test_num:,:];y_test = Y[-test_num:]
    
    ###########  calculate  ######################

    start_time = time.clock()
    rf_result = random_selection(x, num_select, N=500, num_use=int(0.5*d),pctg=pctg, two_sided=False)
    print('rf running time:',time.clock()-start_time)

    start_time = time.clock()
    rank_result,l1,l2,lmax= ranking_selection(x, num_select, N=500, num_use=int(0.5*d),sample_pctg=1, preserve_pctg=pctg)
    print('rank running time:',time.clock()-start_time)
    
    start_time = time.clock()
    lap_score_result = lap_score.lap_score(x)
    lap_score_result= np.argsort(lap_score_result)[:num_select]    #find minimum
    print('lap_score running time:',time.clock()-start_time)
    
    start_time = time.clock()
    SPEC_result = SPEC.spec(x)
    print('SPEC running time:',time.clock()-start_time)
    SPEC_result= np.argsort(SPEC_result)[:num_select]     #find minimum
    
    '''sparse learning based'''
    start_time = time.clock()
    MCFS_W = MCFS.mcfs(x,num_select,**{'n_clusters':num_clusters})
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
    rank_x = x[:,list(rank_result)]
    l1_x = x[:,list(l1)]
    l2_x = x[:,list(l2)]
    lmax_x = x[:,list(lmax)]
    lap_score_x = x[:,list(lap_score_result)]
    SPEC_x = x[:,list(SPEC_result)]
    MCFS_x = x[:,list(MCFS_result)]
#    NDFS_x = x[:,list(NDFS_result)]
#    UDFS_x = x[:,list(UDFS_result)]
    
#    '''[KNN purity NMI dgm0 dgm1], each one is a matrix'''
#    methods = ['rf','rank','lap_score','SPEC','MCFS']
#    for method in methods:
#        if method=='rf':
#            selected_feature = list(rf_result).reverse()
#        elif method=='rank':
#            selected_feature = list(rank_result).reverse()
#        elif method=='lap_score':
#            selected_feature = list(lap_score_result)
#        elif method=='SPEC':
#            selected_feature = list(SPEC_result)
#        else:
#            selected_feature = list(MCFS_result).reverse()
#        
#        if num_select<=50:         # the dimension
#            start_dim = 5; step = 2
#        else:
#            start_dim = 10; step = 5
        
    print('KNN accuracy')
    print('rf', ef.knn_accuracy(x,y,rf_result))
    print('rank', ef.knn_accuracy(x,y,rank_result))
    print('l1', ef.knn_accuracy(x,y,l1))
    print('l2', ef.knn_accuracy(x,y,l2))
    print('lmax', ef.knn_accuracy(x,y,lmax))
    print('lap_score', ef.knn_accuracy(x,y,lap_score_result))
    print('SPEC', ef.knn_accuracy(x,y,SPEC_result))
    print('MCFS',ef.knn_accuracy(x,y,MCFS_result))
#    print('NDFS',ef.knn_accuracy(x_test,y_test,NDFS_result))
#    print('UDFS',ef.knn_accuracy(x_test,y_test,UDFS_result),'\n')  

#    print('connectivity')
#    print('rf', ef.connectivity(x,rf_x,pctg, two_sided))
#    print('rank', ef.connectivity(x,rank_x,pctg, two_sided))
#    print('lap_score', ef.connectivity(x,lap_score_x,pctg, two_sided))
#    print('SPEC', ef.connectivity(x,SPEC_x,pctg, two_sided))
#    print('cut-SPEC', ef.connectivity(x,CSPEC_x,pctg, two_sided))
#    print('MCFS',ef.connectivity(x,MCFS_x,pctg, two_sided))
    
#    print('NDFS',ef.connectivity(x,NDFS_x,pctg, two_sided))
#    print('UDFS',ef.connectivity(x,UDFS_x,pctg, two_sided),'\n')  

    print('purity score | NMI')
    print('origin', ef.purity_score(x,y))
    print('rf', ef.purity_score(rf_x,y))
    print('rank', ef.purity_score(rank_x,y))
    print('lap_score', ef.purity_score(lap_score_x,y))
    print('SPEC', ef.purity_score(SPEC_x,y)  )
    print('MCFS', ef.purity_score(MCFS_x,y))
   
    dgm = ef.compute_dgm(x, t, dim, thresh)
    dgm_rf = ef.compute_dgm(rf_x, t, dim, thresh)
    dgm_rank = ef.compute_dgm(rank_x, t, dim, thresh)
    dgm_l1 = ef.compute_dgm(l1_x, t, dim, thresh)
    dgm_l2 = ef.compute_dgm(l2_x, t, dim, thresh)
    dgm_lmax = ef.compute_dgm(lmax_x, t, dim, thresh)
    dgm_lap_score = ef.compute_dgm(lap_score_x, t, dim, thresh)
    dgm_SPEC = ef.compute_dgm(SPEC_x, t, dim, thresh)
    dgm_MCFS = ef.compute_dgm(MCFS_x, t, dim, thresh)
#    plt.figure()
#    plt.plot(dgm[:,-2:], 'ro')
#    plt.figure()
#    plt.plot(dgm_rf[:,-2:], 'ro')
#    plt.figure()
#    plt.plot(dgm_rank[:,-2:], 'ro')
#    plt.figure()
#    plt.plot(dgm_SPEC[:,-2:], 'ro')
#    plt.figure()
#    plt.plot(dgm_MCFS[:,-2:], 'ro')
    
    print('dgm distance')
    print('rf', ef.dgm_distance(dgm,dgm_rf,'W', dim),'  ',ef.dgm_distance(dgm,dgm_rf,'B', dim))
    print('rank', ef.dgm_distance(dgm,dgm_rank,'W', dim),'  ',ef.dgm_distance(dgm,dgm_rank,'B', dim))
    print('l1', ef.dgm_distance(dgm,dgm_l1,'W', dim),'  ',ef.dgm_distance(dgm,dgm_l1,'B', dim))
    print('l2', ef.dgm_distance(dgm,dgm_l2,'W', dim),'  ',ef.dgm_distance(dgm,dgm_l2,'B', dim))
    print('lmax', ef.dgm_distance(dgm,dgm_lmax,'W', dim),'  ',ef.dgm_distance(dgm,dgm_lmax,'B', dim))
    print('lap_score', ef.dgm_distance(dgm,dgm_lap_score,'W', dim),'  ',ef.dgm_distance(dgm,dgm_lap_score,'B', dim))
    print('SPEC', ef.dgm_distance(dgm,dgm_SPEC,'W', dim),'  ',ef.dgm_distance(dgm,dgm_SPEC,'B', dim))
    print('MCFS', ef.dgm_distance(dgm,dgm_MCFS,'W', dim),'  ',ef.dgm_distance(dgm,dgm_MCFS,'B', dim))

def generate_result(dataset, x,y,num_select, zero_mean=False, N=1000, t=0.6, thresh=0.1):
    if zero_mean == False:
        x = normalize(x,axis=0)
    else:
        x = standardize_feature(x)
        
    n,d = x.shape
    
    if num_select==300:
        start_dim = 20; step = 20
    elif num_select==200:         # the dimension
        start_dim = 20; step = 10
    elif num_select==100:
        start_dim = 10; step = 10
    elif num_select==50:
        start_dim = 10; step = 5
    elif num_select == 20:
        start_dim = 4; step = 2
    else:
        start_dim = 5; step = 1
           
    dimension_list = list(range(start_dim,num_select+1,step))
    
    #########  rank: parameter  preserve_pctg, num_use  #########
    dgm0 = ef.compute_dgm(x, t, 0, thresh)
    D0 = compute_dist(x)
    
    preserve_pctg_list = [0.2,0.4,0.6,0.8,1]   #dimension 0
    num_use_list = [0.1,0.2,0.3,0.4,0.5]    #dimension 1
        
    rank_result = np.zeros([len(preserve_pctg_list),len(num_use_list),8,len(dimension_list)])
    rank_result_l1 = np.zeros([len(preserve_pctg_list),len(num_use_list),8,len(dimension_list)])
    rank_result_l2 = np.zeros([len(preserve_pctg_list),len(num_use_list),8,len(dimension_list)])
    rank_result_lmax = np.zeros([len(preserve_pctg_list),len(num_use_list),8,len(dimension_list)])
    
    for i,preserve_pctg in enumerate(preserve_pctg_list):
        for j,num_use in enumerate(num_use_list):
            print(i,j)
            rank_selected, rank_selected_l1, rank_selected_l2, rank_selected_lmax= ranking_selection(x, num_select, N=N, num_use=int(num_use*d+1),sample_pctg=1, preserve_pctg=preserve_pctg)
            rank_selected = list(rank_selected)[::-1]

            for k,dimension in enumerate(dimension_list):      #performance using different number fo features
                s = rank_selected[:dimension]
                rank_x = x[:,s]
                rank_result[i,j,0,k] = ef.knn_accuracy(x,y,s)
                rank_result[i,j,1,k], rank_result[i,j,2,k] = ef.purity_score(rank_x,y)
               
                dgm_rank0 = ef.compute_dgm(rank_x, t, 0, thresh)
               
                rank_result[i,j,3,k] = ef.dgm_distance(dgm0,dgm_rank0,'W', 0)
                rank_result[i,j,4,k] = ef.dgm_distance(dgm0,dgm_rank0,'B', 0)
                D_rank = compute_dist(rank_x)
                rank_result[i,j,5,k] = ef.dif_dist(D0,D_rank,'l1')
                rank_result[i,j,6,k] = ef.dif_dist(D0,D_rank,'l2')
                rank_result[i,j,7,k] = ef.dif_dist(D0,D_rank,'lmax')
                
                #### 
                s_l1 = rank_selected_l1[:dimension]
                rank_l1_x = x[:,s_l1]
                rank_result_l1[i,j,0,k] = ef.knn_accuracy(x,y,s_l1)
                rank_result_l1[i,j,1,k], rank_result_l1[i,j,2,k] = ef.purity_score(rank_l1_x,y)
               
                dgm_rank_l10 = ef.compute_dgm(rank_l1_x, t, 0, thresh)
               
                rank_result_l1[i,j,3,k] = ef.dgm_distance(dgm0,dgm_rank_l10,'W', 0)
                rank_result_l1[i,j,4,k] = ef.dgm_distance(dgm0,dgm_rank_l10,'B', 0)
                D1 = compute_dist(rank_l1_x)
                
                rank_result_l1[i,j,5,k] = ef.dif_dist(D0,D1,'l1')
                rank_result_l1[i,j,6,k] = ef.dif_dist(D0,D1,'l2')
                rank_result_l1[i,j,7,k] = ef.dif_dist(D0,D1,'lmax')  
                
                ########
                s_l2 = rank_selected_l2[:dimension]
                rank_l2_x = x[:,s_l2]
                rank_result_l2[i,j,0,k] = ef.knn_accuracy(x,y,s_l2)
                rank_result_l2[i,j,1,k], rank_result_l2[i,j,2,k] = ef.purity_score(rank_l2_x,y)
               
                dgm_rank_l20 = ef.compute_dgm(rank_l2_x, t, 0, thresh)
               
                rank_result_l2[i,j,3,k] = ef.dgm_distance(dgm0,dgm_rank_l20,'W', 0)
                rank_result_l2[i,j,4,k] = ef.dgm_distance(dgm0,dgm_rank_l20,'B', 0)
                D2 = compute_dist(rank_l2_x)
                
                rank_result_l2[i,j,5,k] = ef.dif_dist(D0,D2,'l1')
                rank_result_l2[i,j,6,k] = ef.dif_dist(D0,D2,'l2')
                rank_result_l2[i,j,7,k] = ef.dif_dist(D0,D2,'lmax')  
                
                ###########
                s_lmax = rank_selected_lmax[:dimension]
                rank_lmax_x = x[:,s_lmax]
                rank_result_lmax[i,j,0,k] = ef.knn_accuracy(x,y,s_lmax)
                rank_result_lmax[i,j,1,k], rank_result_lmax[i,j,2,k] = ef.purity_score(rank_lmax_x,y)
               
                dgm_rank_lmax0 = ef.compute_dgm(rank_lmax_x, t, 0, thresh)
               
                rank_result_lmax[i,j,3,k] = ef.dgm_distance(dgm0,dgm_rank_lmax0,'W', 0)
                rank_result_lmax[i,j,4,k] = ef.dgm_distance(dgm0,dgm_rank_lmax0,'B', 0)
                D_max = compute_dist(rank_lmax_x)
                
                rank_result_lmax[i,j,5,k] = ef.dif_dist(D0,D_max,'l1')
                rank_result_lmax[i,j,6,k] = ef.dif_dist(D0,D_max,'l2')
                rank_result_lmax[i,j,7,k] = ef.dif_dist(D0,D_max,'lmax')  
                
    np.save('./result/'+dataset+'/rank',rank_result)
    np.save('./result/'+dataset+'/rank_l1',rank_result_l1)
    np.save('./result/'+dataset+'/rank_l2',rank_result_l2)
    np.save('./result/'+dataset+'/rank_lmax',rank_result_lmax)
    
    ########  lap_score  ###########
    lap_score_result = np.zeros([7,len(dimension_list)])
    lap_score_selected = lap_score.lap_score(x)
    lap_score_selected = list(np.argsort(lap_score_selected)[:num_select])    #find minimum
    
    for k,dimension in enumerate(dimension_list):      #performance using different number fo features
        s = lap_score_selected[:dimension]
        lap_score_x = x[:,s]
        lap_score_result[0,k] = ef.knn_accuracy(x,y,s)
        lap_score_result[1,k], lap_score_result[2,k] = ef.purity_score(lap_score_x,y)
       
        dgm_lap_score0 = ef.compute_dgm(lap_score_x, t, 0, thresh)

        D1 = compute_dist(lap_score_x)
       
        lap_score_result[3,k] = ef.dgm_distance(dgm0,dgm_lap_score0,'W', 0)
        lap_score_result[4,k] = ef.dgm_distance(dgm0,dgm_lap_score0,'B', 0)


        lap_score_result[5,k] = ef.dif_dist(D0,D1,'l1')
        lap_score_result[6,k] = ef.dif_dist(D0,D1,'l2')
        lap_score_result[7,k] = ef.dif_dist(D0,D1,'lmax')
    np.save('./result/'+dataset+'/lap_score',lap_score_result)
    
    ########  SPEC  ###########
    SPEC_result = np.zeros([7,len(dimension_list)])
    SPEC_selected = SPEC.spec(x)
    SPEC_selected = list(np.argsort(SPEC_selected)[:num_select])    #find minimum
    
    for k,dimension in enumerate(dimension_list):      #performance using different number fo features
        s = SPEC_selected[:dimension]
        SPEC_x = x[:,s]
        SPEC_result[0,k] = ef.knn_accuracy(x,y,s)
        SPEC_result[1,k], SPEC_result[2,k] = ef.purity_score(SPEC_x,y)
       
        dgm_SPEC0 = ef.compute_dgm(SPEC_x, t, 0, thresh)
     
        SPEC_result[3,k] = ef.dgm_distance(dgm0,dgm_SPEC0,'W', 0)
        SPEC_result[4,k] = ef.dgm_distance(dgm0,dgm_SPEC0,'B', 0)
  
        D1 = compute_dist(SPEC_x)
        
        SPEC_result[5,k] = ef.dif_dist(D0,D1,'l1')
        SPEC_result[6,k] = ef.dif_dist(D0,D1,'l2')
        SPEC_result[7,k] = ef.dif_dist(D0,D1,'lmax')

    np.save('./result/'+dataset+'/SPEC',SPEC_result)
    
    #######  MCFS  parameter: num_clusters  ##############   
    num_clusters_list = [5,10,20,30]     
    MCFS_result = np.zeros([len(num_clusters_list),7,len(dimension_list)])
    for i,num_clusters in enumerate(num_clusters_list):
        MCFS_W = MCFS.mcfs(x,num_select,**{'n_clusters':num_clusters})
        MCFS_selected = [np.max(np.abs(x)) for x in MCFS_W]     #find maximum
        MCFS_selected= np.argsort(MCFS_selected)[-num_select:]
        MCFS_selected = list(MCFS_selected)[::-1]
        for k,dimension in enumerate(dimension_list):      #performance using different number fo features
            s = MCFS_selected[:dimension]
            MCFS_x = x[:,s]
            MCFS_result[i,0,k] = ef.knn_accuracy(x,y,s)
            MCFS_result[i,1,k], MCFS_result[i,2,k] = ef.purity_score(MCFS_x,y)
           
            dgm_MCFS0 = ef.compute_dgm(MCFS_x, t, 0, thresh)
           
            MCFS_result[i,3,k] = ef.dgm_distance(dgm0,dgm_MCFS0,'W', 0)
            MCFS_result[i,4,k] = ef.dgm_distance(dgm0,dgm_MCFS0,'B', 0)
            D1 = compute_dist(MCFS_x)
            
            MCFS_result[i,5,k] = ef.dif_dist(D0,D1,'l1')
            MCFS_result[i,6,k] = ef.dif_dist(D0,D1,'l2')
            MCFS_result[i,7,k] = ef.dif_dist(D0,D1,'lmax')

        
    np.save('./result/'+dataset+'/MCFS',MCFS_result)   
    
    return rank_result, rank_result_l1, rank_result_l2,rank_result_lmax,lap_score_result, SPEC_result, MCFS_result

def generate_result_dist(dataset, x,y,num_select, zero_mean=False, N=1000, t=0.6, thresh=0.1):
    if zero_mean == False:
        x = normalize(x,axis=0)
    else:
        x = standardize_feature(x)
        
    n,d = x.shape
    
    if num_select==300:
        start_dim = 20; step = 20
    elif num_select==200:         # the dimension
        start_dim = 20; step = 10
    elif num_select==100:
        start_dim = 10; step = 10
    elif num_select==50:
        start_dim = 10; step = 5
    elif num_select == 20:
        start_dim = 4; step = 2
    else:
        start_dim = 5; step = 1
           
    dimension_list = list(range(start_dim,num_select+1,step))
    
    #########  rank: parameter  preserve_pctg, num_use  #########
    D0 = compute_dist(x)
    
    preserve_pctg_list = [0.2,0.4,0.6,0.8,1]   #dimension 0
    num_use_list = [0.1,0.2,0.3,0.4,0.5]    #dimension 1
        
    rank_result = np.zeros([len(preserve_pctg_list),len(num_use_list),7,len(dimension_list)])
    rank_result_l1 = np.zeros([len(preserve_pctg_list),len(num_use_list),7,len(dimension_list)])
    rank_result_l2 = np.zeros([len(preserve_pctg_list),len(num_use_list),7,len(dimension_list)])
    rank_result_lmax = np.zeros([len(preserve_pctg_list),len(num_use_list),7,len(dimension_list)])
    
    for i,preserve_pctg in enumerate(preserve_pctg_list):
        for j,num_use in enumerate(num_use_list):
            print(i,j)
            rank_selected, rank_selected_l1, rank_selected_l2, rank_selected_lmax= ranking_selection(x, num_select, N=N, num_use=int(num_use*d+1),sample_pctg=1, preserve_pctg=preserve_pctg)
            rank_selected = list(rank_selected)[::-1]

            for k,dimension in enumerate(dimension_list):      #performance using different number fo features
                s = rank_selected[:dimension]
                rank_x = x[:,s]
                D_rank = compute_dist(rank_x)
                rank_result[i,j,0,k] = ef.dif_dist(D0,D_rank,'l1')
                rank_result[i,j,1,k] = ef.dif_dist(D0,D_rank,'l2')
                rank_result[i,j,2,k] = ef.dif_dist(D0,D_rank,'lmax')
                
                s_l1 = rank_selected_l1[:dimension]
                rank_l1_x = x[:,s_l1]
                D1 = compute_dist(rank_l1_x)
                
                rank_result_l1[i,j,0,k] = ef.dif_dist(D0,D1,'l1')
                rank_result_l1[i,j,1,k] = ef.dif_dist(D0,D1,'l2')
                rank_result_l1[i,j,2,k] = ef.dif_dist(D0,D1,'lmax')               

                s_l2 = rank_selected_l2[:dimension]
                rank_l2_x = x[:,s_l2]
                D2 = compute_dist(rank_l2_x)
                
                rank_result_l2[i,j,0,k] = ef.dif_dist(D0,D2,'l1')
                rank_result_l2[i,j,1,k] = ef.dif_dist(D0,D2,'l2')
                rank_result_l2[i,j,2,k] = ef.dif_dist(D0,D2,'lmax')  
                
                s_lmax = rank_selected_lmax[:dimension]
                rank_lmax_x = x[:,s_lmax]
                D_max = compute_dist(rank_lmax_x)
                
                rank_result_lmax[i,j,0,k] = ef.dif_dist(D0,D_max,'l1')
                rank_result_lmax[i,j,1,k] = ef.dif_dist(D0,D_max,'l2')
                rank_result_lmax[i,j,2,k] = ef.dif_dist(D0,D_max,'lmax')                 

    
    np.save('./result/'+dataset+'/rank_dist',rank_result)
    np.save('./result/'+dataset+'/rank_l1_dist',rank_result_l1)
    np.save('./result/'+dataset+'/rank_l2_dist',rank_result_l2)
    np.save('./result/'+dataset+'/rank_lmax_dist',rank_result_lmax)
    
    ########  lap_score  ###########
    lap_score_result = np.zeros([7,len(dimension_list)])
    lap_score_selected = lap_score.lap_score(x)
    lap_score_selected = list(np.argsort(lap_score_selected)[:num_select])    #find minimum
    
    for k,dimension in enumerate(dimension_list):      #performance using different number fo features
        s = lap_score_selected[:dimension]
        lap_score_x = x[:,s]
        D1 = compute_dist(lap_score_x)
        
        lap_score_result[0,k] = ef.dif_dist(D0,D1,'l1')
        lap_score_result[1,k] = ef.dif_dist(D0,D1,'l2')
        lap_score_result[2,k] = ef.dif_dist(D0,D1,'lmax')

    np.save('./result/'+dataset+'/lap_score_dist',lap_score_result)
    
    ########  SPEC  ###########
    SPEC_result = np.zeros([7,len(dimension_list)])
    SPEC_selected = SPEC.spec(x)
    SPEC_selected = list(np.argsort(SPEC_selected)[:num_select])    #find minimum
    
    for k,dimension in enumerate(dimension_list):      #performance using different number fo features
        s = SPEC_selected[:dimension]
        SPEC_x = x[:,s]
        D1 = compute_dist(SPEC_x)
        
        SPEC_result[0,k] = ef.dif_dist(D0,D1,'l1')
        SPEC_result[1,k] = ef.dif_dist(D0,D1,'l2')
        SPEC_result[2,k] = ef.dif_dist(D0,D1,'lmax')

    np.save('./result/'+dataset+'/SPEC_dist',SPEC_result)
    
    #######  MCFS  parameter: num_clusters  ##############   
    num_clusters_list = [5,10,20,30]     
    MCFS_result = np.zeros([len(num_clusters_list),7,len(dimension_list)])
    for i,num_clusters in enumerate(num_clusters_list):
        MCFS_W = MCFS.mcfs(x,num_select,**{'n_clusters':num_clusters})
        MCFS_selected = [np.max(np.abs(x)) for x in MCFS_W]     #find maximum
        MCFS_selected= np.argsort(MCFS_selected)[-num_select:]
        MCFS_selected = list(MCFS_selected)[::-1]
        for k,dimension in enumerate(dimension_list):      #performance using different number fo features
            s = MCFS_selected[:dimension]
            MCFS_x = x[:,s]
            D1 = compute_dist(MCFS_x)
            
            MCFS_result[i,0,k] = ef.dif_dist(D0,D1,'l1')
            MCFS_result[i,1,k] = ef.dif_dist(D0,D1,'l2')
            MCFS_result[i,2,k] = ef.dif_dist(D0,D1,'lmax')
           
        
    np.save('./result/'+dataset+'/MCFS_dist',MCFS_result)   
    
    return rank_result, rank_result_l1, rank_result_l2,rank_result_lmax,lap_score_result, SPEC_result, MCFS_result

def generate_original(x,y,zero_mean=False):
    if zero_mean == False:
        x = normalize(x,axis=0)
    else:
        x = standardize_feature(x)
        
    n,d = x.shape
    
    original_result = np.zeros([3])   #1:KNN 2： PURITY 3：NMI
    s = list(range(d))
    original_result[0] = ef.knn_accuracy(x,y,s)
    original_result[1], original_result[2]= ef.purity_score(x,y)
    np.save('./result/'+dataset+'/original',original_result)
    
#########  datasets  from package##############
datasets = os.listdir('D:/Anaconda3/Lib/site-packages/skfeature/data')
#datasets = ['ALLAML.mat']
shape = []
num_class = []
for dataset in datasets:
    data = scipy.io.loadmat('D:/Anaconda3/Lib/site-packages/skfeature/data/'+dataset)
    X = data['X']
    y = data['Y']
    shape.append(X.shape)
    c = len(np.unique(y))
    num_class.append(c)
    if X.shape[0]<1000 and X.shape[1]<10000:
        create_folder(dataset)    #path '.result/dataset'
        n,d = X.shape
        print(dataset)
        print(X.shape,len(np.unique(y)))
        if d>500:
            target_dim = 300
        elif d>200:
            target_dim = 200
        elif d>100:
            target_dim = 100
        elif d>50:
            target_dim = 50
        elif d>20:
            target_dim = 20
        else:
            target_dim = d
            
        np.random.seed(1993)
#        compare_methods(X,y,num_select=target_dim,pctg=0.5, num_clusters=5, zero_mean=False,dim=0,t=0.5,thresh=0.1)
        generate_original(X,y)
        rank_result, rank_result_l1, rank_result_l2,rank_result_lmax,lap_score_result, SPEC_result, MCFS_result = generate_result_dist(dataset, X,y,num_select=target_dim, zero_mean=False, N = 1000, t=0.5, thresh=0.1)
        print('---------------------------------')
    
########  datasets from UCI  ##############
print('+++++++++++++UCI data+++++++++++++++++++\n')
datasets = os.listdir('..\data')
#datasets = ['breast-cancer-wisconsin.data']
data = []
label = []
for k,dataset in enumerate(datasets):
    data1 = pd.read_table('..\data\\'+dataset,sep=",", header= None, error_bad_lines=False)
    data1 = data1.replace('?',np.nan).dropna()

    if k in [0,1,2,7]:
        data1 = data1.iloc[:,1:]
        
    if k in [3,4]:
        data1 = pd.read_table('..\data\\'+dataset,sep=",", error_bad_lines=False)
        
    if dataset != 'wdbc.data':
        y = data1.iloc[:,-1]
        X = data1.iloc[:,:-1]
    else:
        y=data1.iloc[:,0]
        X=data1.loc[:,data1.columns != 1]

    if dataset =='drug_consumption.data':
        y = data1.iloc[:,12]
        data1 = data1.select_dtypes(include=['number'])
        X = data1
    
    X = X.loc[:, (X != X.iloc[0]).any()] 
    X = np.array(X,'float64')
    y = np.array(y)

    print(X.shape)
    n,d = X.shape
    
    data.append(X)
    label.append(y)
    
    a = compute_dist(X)
    
    num_class = len(np.unique(y))
    if d>500:
        target_dim = 300
    elif d>200:
        target_dim = 200
    elif d>100:
        target_dim = 100
    elif d>50:
        target_dim = 50
    elif d>20:
        target_dim = 20
    else:
        target_dim = d
    if n<1000:
        print(dataset)
        create_folder(dataset)
        np.random.seed(1993)
#        compare_methods(X,y,num_select=target_dim,pctg=0.5, num_clusters=num_class, zero_mean=False,dim=0,t=0.5,thresh=0.1)
        generate_original(X,y)
        rank_result, rank_result_l1, rank_result_l2,rank_result_lmax,lap_score_result, SPEC_result, MCFS_result = generate_result_dist(dataset, X,y,num_select=target_dim, zero_mean=False, N = 1000, t=0.5, thresh=0.1)
        print('--------------------------------')

'''
t=0.5, 0.7  thresh=0.05  dim=0   good for many datasets

'''




