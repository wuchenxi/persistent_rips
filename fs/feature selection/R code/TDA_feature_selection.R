library(TDA)
library(mlbench)
library(Rtsne)
library(tsne)
library(ggplot2)
library(glmnet)
library(lars)
library(IDmining)

setwd('C:/Users/Xiaoyun Li/Desktop/PhD/Papers/Topological Data Analysis/feature selection/R code')

set.seed(1993)

rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}

rep.col<-function(x,n){
  matrix(rep(x,each=n),ncol=n, byrow=TRUE)
}

compute_dist=function(X){
  n=nrow(X);d=ncol(X)
  A=X%*%t(X)
  d=diag(A)
  B=rep.row(d,n)
  return(sqrt(B+t(B)-2*A))
}

simulate_regression=function(n,d,small_parameters,irrelavents,sigma=1){
  X_mean = runif(d)*10
  X = matrix(rnorm(n*d,0,1),n,d)+X_mean
  bad_dims = sample(d,irrelavents,replace = F)
  X[,bad_dims] = matrix(rnorm(n*irrelavents,0,0.04),n,irrelavents)
  
  para = runif(d,-2,5)
  eps = runif(small_parameters,-0.1,0.1)
  para[(d-small_parameters+1):d] = eps
  noise = rnorm(n,0,sigma)
  y = X%*%para+noise
  return(list(X=X, y=y, true_beta=para))
}

simulate_ball=function(n,d,d_noise,r=5,sigma=1){
  X=matrix(0,n,d+2+d_noise)
  for(k in 1:n){
    x=c()
    remain=r^2
    for(i in 1:d){
      x1=runif(1,-sqrt(remain),sqrt(remain))
      remain=r^2-x1^2
      x=c(x,x1)
    }
    remain=3^2
    for(i in 1:2){
      x1=runif(1,-sqrt(remain),sqrt(remain))
      remain=r^2-x1^2
      x=c(x,x1)
    }
    x=c(x,rnorm(d_noise,0,sigma))
    X[k,]=x
  }
  return(X)
}

TDA_selection_with_y=function(X,target_dim,votes_num=3,dgm_dim=1){
  n=nrow(X);d=ncol(X)
  if (is.null(colnames(X))){
    colnames(X)=1:d
  }
  
  throw_list=c()
  for(k in 1:(d-target_dim)){
    cat('throwing ',k,'-th feature\n')
    feature_set=colnames(X)
    print(feature_set)
    feature_set=feature_set[-c(1,2)]
    yy=X[,1:2]
    dist_matrix=compute_dist(X)
    if(votes_num!=1){
      d_range=quantile(dist_matrix,c(0.2,0.4))
      d_list=seq(d_range[1],d_range[2],(d_range[2]-d_range[1])/(votes_num-1))
    }
    else{
      d_list=quantile(dist_matrix,c(0.2))
    }
    
    all_votes=c()
    for(d in d_list){
      print(d)
      dgm_origin=ripsDiag(X,dgm_dim,d,library='GUDHI')
      dgm1 = matrix(as.numeric(dgm_origin$diagram),nrow(dgm_origin$diagram),3)
      a=dgm1[which(dgm1[,1]!=0),]
      death_minus_birth=a[,3]-a[,2]
      # eps=quantile(death_minus_birth,0.9)
      eps=max(death_minus_birth)*0.3
      origin_rank=length(which(dgm1[,1]!=0&dgm1[,3]-dgm1[,2]>=eps))
      print(origin_rank)
      
      reduced_rank=c()
      for(feature in feature_set){
        X_reduced = X[,which(colnames(X)!=feature)]
        X_reduced = cbind(yy,X_reduced)
        dgm_reduced=ripsDiag(X_reduced,dgm_dim,d,library='GUDHI')
        dgm2 = matrix(as.numeric(dgm_reduced$diagram),nrow(dgm_reduced$diagram),3)
        reduced_rank=c(reduced_rank,length(which(dgm2[,1]!=0&dgm2[,3]-dgm2[,2]>=eps)))
      }
      throw_feature_d=feature_set[which.max(reduced_rank[-c(1,2)])+2]
      print(reduced_rank)
      all_votes=c(all_votes,throw_feature_d)
    }
    print(all_votes)
    throw_feature=sample(names(which.max(table(all_votes))),1)    #if multiple choice, random choose one
    throw_list=c(throw_list,throw_feature)     #update throw list
    X = X[,which(colnames(X)!=throw_feature)]
  }
  
  return(list(throw_list=throw_list,selected=colnames(X)))
}

TDA_selection=function(X,target_dim,votes_num=3,dgm_dim=1){
  n=nrow(X);d=ncol(X)
  if (is.null(colnames(X))){
    colnames(X)=1:d
  }
  
  throw_list=c()
  for(k in 1:(d-target_dim)){
    cat('throwing ',k,'-th feature\n')
    feature_set=colnames(X)
    dist_matrix=compute_dist(X)
    if(votes_num!=1){
      d_range=quantile(dist_matrix,c(0.2,0.4))
      d_list=seq(d_range[1],d_range[2],(d_range[2]-d_range[1])/(votes_num-1))
    }
    else{
      d_list=quantile(dist_matrix,c(0.2))
    }

    all_votes=c()
    for(d in d_list){
      print(d)
      dgm_origin=ripsDiag(X,dgm_dim,d,library='GUDHI')
      dgm1 = matrix(as.numeric(dgm_origin$diagram),nrow(dgm_origin$diagram),3)
      a=dgm1[which(dgm1[,1]!=0),]
      death_minus_birth=a[,3]-a[,2]
      # eps=quantile(death_minus_birth,0.9)
      eps=max(death_minus_birth)*0.3
      origin_rank=length(which(dgm1[,1]!=0&dgm1[,3]-dgm1[,2]>=eps))
      print(origin_rank)
      
      reduced_rank=c()
      for(feature in feature_set){
        X_reduced = X[,which(colnames(X)!=feature)]
        dgm_reduced=ripsDiag(X_reduced,dgm_dim,d,library='GUDHI')
        dgm2 = matrix(as.numeric(dgm_reduced$diagram),nrow(dgm_reduced$diagram),3)
        reduced_rank=c(reduced_rank,length(which(dgm2[,1]!=0&dgm2[,3]-dgm2[,2]>=eps)))
      }
      throw_feature_d=feature_set[which.max(reduced_rank)]
      print(reduced_rank)
      all_votes=c(all_votes,throw_feature_d)
    }
    print(all_votes)
    throw_feature=sample(names(which.max(table(all_votes))),1)    #if multiple choice, random choose one
    throw_list=c(throw_list,throw_feature)     #update throw list
    X = X[,which(colnames(X)!=throw_feature)]
  }
  
  return(list(throw_list=throw_list,selected=colnames(X)))
}

backward_distance_selection=function(X, target_dim, pctg=0.1,sample_weight = rep(1,nrow(X))){
  n=nrow(X);d=ncol(X)
  if (is.null(colnames(X))){
    colnames(X)=1:d
  }
  
  throw_list=c()
  for(k in 1:(d-target_dim)){
    cat('throwing ',k,'-th feature\n')
    feature_set=colnames(X)
    dist_matrix=compute_dist(X)
    # diag(dist_matrix)=Inf
    threshold = quantile(dist_matrix,1-pctg)
    original_connection = (dist_matrix>threshold)
    
    score=c()
    for(feature in feature_set){
      X_reduced = X[,which(colnames(X)!=feature)]
      reduced_dist=compute_dist(X_reduced)
      # diag(reduced_dist)=Inf
      threshold = quantile(reduced_dist,1-pctg)
      reduced_connection = (reduced_dist>threshold)
      still_connected = (original_connection & reduced_connection)
      sample_persistence = apply(still_connected,1,sum)
      score=c(score,sum(sample_persistence*sample_weight))
    }
    names(score) = feature_set
    print(score)
    throw_feature=feature_set[which.min(score)]
    throw_list=c(throw_list,throw_feature)     #update throw list
    X = X[,which(colnames(X)!=throw_feature)]
  }
  return(list(throw_list=throw_list,selected=colnames(X)))
}

# n=50;d=10
# small_parameters=d*0.4
# irrelavents=d*0.5
# data = simulate_regression(n,d,small_parameters,irrelavents)

###########  boston housing data  ############
data(BostonHousing)
housing_data <- BostonHousing

housing_data=data.matrix(housing_data)
X = housing_data[,1:13]; y = housing_data[,14]
X = X[, !(colnames(X) %in% c('chas','rad'))]
n=nrow(X);d=ncol(X)
X=(X-rep.row(apply(X,2,mean),n))/rep.row(apply(X,2,sd),n)

ypi = (y-min(y))/(max(y)-min(y))*2*pi
y_feature = cbind(sin(ypi),cos(ypi))
y_feature=(y_feature-rep.row(apply(y_feature,2,mean),n))/rep.row(apply(y_feature,2,sd),n)
colnames(y_feature)=c('sin_y','cos_y')
X_combine = cbind(y_feature,X)

n = nrow(X); d = ncol(X)

dgm_origin=ripsDiag(X,1,2,library='GUDHI')$diagram

a=ripsFiltration(X,1,2)

result=TDA_selection_with_y(X_combine,8,votes_num=1,dgm_dim=1)
result=TDA_selection(X,8,votes_num=1,dgm_dim=1)

result$throw_list
result$selected

X_selected=X[,which(colnames(X) %in% result$selected)]
reg=lm(y~X)
summary(reg)

########### USELESS plots  #######
# a=Rtsne(X)
# qplot(X1,X2, data=data.frame(a$Y),colour=y) + scale_colour_gradient(low="blue", high="red")
# 
# a=Rtsne(X_selected)
# qplot(X1,X2, data=data.frame(a$Y),colour=y) + scale_colour_gradient(low="blue", high="red")
# 
# a=Rtsne(X[,1:7])
# qplot(X1,X2, data=data.frame(a$Y),colour=y) + scale_colour_gradient(low="blue", high="red")

#################  ball  ############
ball=simulate_ball(500,3,5,r=5,sigma=1)
n=nrow(ball);d=ncol(ball)
ball=cbind(ball,rep.col(t(ball[,10]),2))
ball=(ball-rep.row(apply(ball,2,mean),n))/rep.row(apply(ball,2,sd),n)

##########  Backward distance selection  ########
BDS_result = backward_distance_selection(X,8,0.1)
BDS_result$throw_list
BDS_result$selected

BDS_result = backward_distance_selection(ball,5,0.01)
BDS_result$throw_list
BDS_result$selected

# result=TDA_selection(ball,5,votes_num=1,dgm_dim=1)
# result$throw_list
# result$selected


# ############### other methods  ##############
# LARS=lars(X,y,normalize = F)
# print(LARS)
# 
# 
# source("evaluation_functions.R")
# entropy(X)
# for (i in colnames(X)){
#   cat(i,'  ',entropy(X[,which(colnames(X)!=i)]), '\n')
# }
# entropy(X_selected)

