pair_dist=function(x1,x2,Mm){
  d=sum((x1-x2)^2/Mm^2)
  return(d)
}

entropy=function(X){      #n samples  d features
  n=nrow(X);d=ncol(X)
  Mm=apply(X,2,max)-apply(X,2,min)
  E=0
  D=matrix(0,n,n)
  for(i in 1:n){
    for(j in 1:n){
      D[i,j]=pair_dist(X[i,],X[j,],Mm)
    }
  }
  alpha=-log(0.5)/mean(D)
  sim=exp(-alpha*D)
  for(i in 1:n){
    for(j in 1:n){
      if(i!=j){
        E=E-(sim[i,j]*log(sim[i,j])+(1-sim[i,j])*log(1-sim[i,j]))/n/(n-1)
      }
    }
  }
  return(E)
}