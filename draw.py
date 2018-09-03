# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:21:52 2018

@author: Xiaoyun Li
"""

import matplotlib.pyplot as plt
import numpy as np
import os

datasets = os.listdir('./result')
#datasets=['ALLAML.mat']
for dataset in datasets:

    print(dataset)
    rank_result = np.load('./result/'+dataset+'/rank.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    l1_result = np.load('./result/'+dataset+'/rank_l1.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    l2_result = np.load('./result/'+dataset+'/rank_l2.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    lmax_result = np.load('./result/'+dataset+'/rank_lmax.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    lap_score_result = np.load('./result/'+dataset+'/lap_score.npy')
    SPEC_result = np.load('./result/'+dataset+'/SPEC.npy')
    MCFS_result = np.load('./result/'+dataset+'/MCFS.npy')
    
    rank_result_dist = np.load('./result/'+dataset+'/rank_dist.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    l1_result_dist = np.load('./result/'+dataset+'/rank_l1_dist.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    l2_result_dist = np.load('./result/'+dataset+'/rank_l2_dist.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    lmax_result_dist = np.load('./result/'+dataset+'/rank_lmax_dist.npy')     #dim 0 :preserve_pctg  dim 1: num_use
    lap_score_result_dist = np.load('./result/'+dataset+'/lap_score_dist.npy')
    SPEC_result_dist = np.load('./result/'+dataset+'/SPEC_dist.npy')
    MCFS_result_dist = np.load('./result/'+dataset+'/MCFS_dist.npy')
    
    original_result = np.load('./result/'+dataset+'/original.npy')
    preserve_pctg_list = [0.2,0.4,0.6,0.8,1]   #dimension 0
    num_use_list = [0.1,0.2,0.3,0.4,0.5]    #dimension 1
    dimension_len = rank_result_dist.shape[3]
    
    bench = 0
    ######## find best rank ############
    a = np.zeros([rank_result.shape[0],rank_result.shape[1]])
    a1 = np.zeros([rank_result.shape[0],rank_result.shape[1]])
    a2 = np.zeros([rank_result.shape[0],rank_result.shape[1]])
    amax = np.zeros([rank_result.shape[0],rank_result.shape[1]])
    for p in range(rank_result.shape[0]):
        for q in range(rank_result.shape[1]): 
           a[p,q] = rank_result[p,q,bench,-1]
           a1[p,q] = l1_result[p,q,bench,-1]
           a2[p,q] = l2_result[p,q,bench,-1]
           amax[p,q] = lmax_result[p,q,bench,-1]
    P,Q = np.unravel_index(a.argmin(), a.shape)
    P1,Q1 = np.unravel_index(a1.argmin(), a.shape)
    P2,Q2 = np.unravel_index(a2.argmin(), a.shape)
    Pmax,Qmax = np.unravel_index(amax.argmin(), a.shape)

    ######## find best MCFS ############
    a = np.zeros([MCFS_result.shape[0]])
    for p in range(MCFS_result.shape[0]):
           a[p] = MCFS_result[p,bench,-1]
    K = np.argmax(a)
    
    ########  KNN  ########
    plt.figure(figsize=(12,12))
    plt.subplot(3,3,1)
    plt.plot(rank_result[P,Q,0,:],'c.-')
    plt.plot(l1_result[P1,Q1,0,:])
    plt.plot(l2_result[P2,Q2,0,:])
    plt.plot(lmax_result[Pmax,Qmax,0,:])
    plt.hlines(original_result[0],0,dimension_len)
    
    plt.plot(lap_score_result[0,:],'r--')
    plt.plot(SPEC_result[0,:],'b--')
    plt.plot(MCFS_result[K,0,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' KNN Accuracy')

    ########  purity  ########
#    plt.figure()
    plt.subplot(3,3,2)
    plt.plot(rank_result[P,Q,1,:],'c.-')
    plt.plot(l1_result[P1,Q1,1,:])
    plt.plot(l2_result[P2,Q2,1,:])
    plt.plot(lmax_result[Pmax,Qmax,1,:])  
    plt.hlines(original_result[1],0,dimension_len)
    plt.plot(lap_score_result[1,:],'r--')
    plt.plot(SPEC_result[1,:],'b--')
    plt.plot(MCFS_result[K,1,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' Purity score')

    ########  NMI  ########
#    plt.figure()
    plt.subplot(3,3,3)
    plt.plot(rank_result[P,Q,2,:],'c.-')
    plt.plot(l1_result[P1,Q1,2,:])
    plt.plot(l2_result[P2,Q2,2,:])
    plt.plot(lmax_result[Pmax,Qmax,2,:])
    plt.hlines(original_result[2],0,dimension_len)        
    plt.plot(lap_score_result[2,:],'r--')
    plt.plot(SPEC_result[2,:],'b--')
    plt.plot(MCFS_result[K,2,:],'g--')
    
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' NMI')

    ########  Wasserstein distance dim0  ########
#    plt.figure()
    plt.subplot(3,3,4)
    plt.plot(rank_result[P,Q,3,:],'c.-')
    plt.plot(l1_result[P1,Q1,3,:])
    plt.plot(l2_result[P2,Q2,3,:])
    plt.plot(lmax_result[Pmax,Qmax,3,:])
            
    plt.plot(lap_score_result[3,:],'r--')
    plt.plot(SPEC_result[3,:],'b--')
    plt.plot(MCFS_result[K,3,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' Wasserstein distance dim0')

    ########  Bottleneck distance dim0  ########
#    plt.figure()
    plt.subplot(3,3,5)
    plt.plot(rank_result[P,Q,4,:],'c.-')
    plt.plot(l1_result[P1,Q1,4,:])
    plt.plot(l2_result[P2,Q2,4,:])
    plt.plot(lmax_result[Pmax,Qmax,4,:])
            
    plt.plot(lap_score_result[4,:],'r--')
    plt.plot(SPEC_result[4,:],'b--')
    plt.plot(MCFS_result[K,4,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' Bottleneck distance dim0')

    ########### l1 norm  ##########
    plt.subplot(3,3,6)
    plt.plot(rank_result_dist[P,Q,0,:],'c.-')
    plt.plot(l1_result_dist[P1,Q1,0,:])
    plt.plot(l2_result_dist[P2,Q2,0,:])
    plt.plot(lmax_result_dist[Pmax,Qmax,0,:])
            
    plt.plot(lap_score_result_dist[0,:],'r--')
    plt.plot(SPEC_result_dist[0,:],'b--')
    plt.plot(MCFS_result_dist[K,0,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' L1 norm')

    ##########  l2 norm ############
    plt.subplot(3,3,7)
    plt.plot(rank_result_dist[P,Q,1,:],'c.-')
    plt.plot(l1_result_dist[P1,Q1,1,:])
    plt.plot(l2_result_dist[P2,Q2,1,:])
    plt.plot(lmax_result_dist[Pmax,Qmax,1,:])
            
    plt.plot(lap_score_result_dist[1,:],'r--')
    plt.plot(SPEC_result_dist[1,:],'b--')
    plt.plot(MCFS_result_dist[K,1,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' L2 norm')

    ##########  lmax norm ############
    plt.subplot(3,3,8)
    plt.plot(rank_result_dist[P,Q,2,:],'c.-')
    plt.plot(l1_result_dist[P1,Q1,2,:])
    plt.plot(l2_result_dist[P2,Q2,2,:])
    plt.plot(lmax_result_dist[Pmax,Qmax,2,:])
            
    plt.plot(lap_score_result_dist[2,:],'r--')
    plt.plot(SPEC_result_dist[2,:],'b--')
    plt.plot(MCFS_result_dist[K,2,:],'g--')
    plt.legend(['rank','l1','l2','lmax'])
    plt.title(dataset+' Lmax norm')