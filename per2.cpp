//Experiment 3: Classification of Reddit-5k datasets

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

/***************Persistent Homology*************************/
const int maxbd=3;
const int maxdim=1;
const int npts=100;
typedef struct fc{int dim; int n; int bd[maxbd];} face;
typedef struct col{int id; std::vector<int> bd;} column;
typedef std::vector<face> cplx;
typedef std::vector<col> mat;

int init(cplx& X, mat& M, std::vector<double>& F, double high){
  int N=X.size();
  M.clear();
  for(int i=0;i<N;i++){
    if(F[i]<high)
      M.push_back({i, {}});
  }
  std::sort(M.begin(), M.end(), [&F](col c1, col c2){return F[c1.id]<F[c2.id];});
  std::vector<int> t(N);
  for(int i=0;i<M.size();i++){t[M[i].id]=i;}
  for(int i=0;i<M.size();i++){
    for(int j=0;j<X[M[i].id].n;j++){
      M[i].bd.push_back(t[X[M[i].id].bd[j]]);
    }
    std::sort(M[i].bd.begin(), M[i].bd.end());
  }
  return 0;
}

int merge(std::vector<int>& v1, std::vector<int>& v2, std::vector<int>& r){
  std::vector<int>::iterator i=v1.begin();
  std::vector<int>::iterator j=v2.begin();
  while(i<v1.end() || j<v2.end()){
    if(i<v1.end() && (j==v2.end() || *i<*j)){r.push_back(*i);i++;}
    else if(j<v2.end() && (i==v1.end() || *j<*i)){r.push_back(*j);j++;}
    else {i++; j++;}
  }
  return 0;
}

int reduce(mat& M){
  int N=M.size();
  int low;
  std::vector<int> L(N,-1);
  for(int i=0;i<N;i++){
    while(M[i].bd.size()&&((low=L[M[i].bd.back()])!=-1)){
      std::vector<int> R;
      merge(M[i].bd, M[low].bd, R);
      M[i].bd=R;
    }
    if(M[i].bd.size()){
      L[M[i].bd.back()]=i;
    }
  }
  return 0;
}


double x[npts];
double y[npts];

int print_barcode(double high, double minlen, mat& M, cplx& X, std::vector<double>& F){
  int N=M.size();
  double val_0[npts];
  double val_1[npts];
  for(int i=0;i<npts;i++){
    val_0[i]=0;
    val_1[i]=0;
  }
  std::vector<char> L(N, 0);
  for(int i=0;i<N;i++){
    if(M[i].bd.size()){
      int low=M[i].bd.back();
      L[low]=1;
      if(X[M[low].id].dim<=maxdim){
	double birth=F[M[low].id];
	double death=F[M[i].id];
	if(death-birth>=minlen){
	  double curx=birth+death;
	  double cury=death-birth;
	  if(cury<0.1)cury=log(cury/0.1)+0.1;
	  if(X[M[low].id].dim==0)
	    for(int j=0;j<npts;j++)
	      val_0[j]+=exp(-((x[j]-curx)*(x[j]-curx)+(y[j]-cury)*(y[j]-cury)));
	  else
	    for(int j=0;j<npts;j++)
	      val_1[j]+=exp(-((x[j]-curx)*(x[j]-curx)+(y[j]-cury)*(y[j]-cury)));
	}
      }
    }
  }
  for(int i=0;i<N;i++){
    if(L[i]==0 && M[i].bd.size()==0 && X[M[i].id].dim<=maxdim){
      double birth=F[M[i].id];
      double death=high;
      if(death-birth>=minlen){
	
	double curx=birth+death;
	double cury=death-birth;
	if(cury<0.1)cury=log(cury/0.1)+0.1;
	if(X[M[i].id].dim==0)
	  for(int j=0;j<npts;j++)
	    val_0[j]+=exp(-((x[j]-curx)*(x[j]-curx)+(y[j]-cury)*(y[j]-cury)));
	else
	  for(int j=0;j<npts;j++)
	    val_1[j]+=exp(-((x[j]-curx)*(x[j]-curx)+(y[j]-cury)*(y[j]-cury)));
      }
    }
  }

  for(int i=0;i<npts;i++)printf(" %lf", val_0[i]);
  for(int i=0;i<npts;i++)printf(" %lf", val_1[i]);
  return 0;
}

int PH0(double high, double minlen, cplx X, std::vector<double>& F){
  mat M;
  init(X, M, F, high);
  reduce(M);
  print_barcode(high, minlen, M, X, F);
  return 0;
}

int PH1(double high, double minlen, cplx X, std::vector<double>& F,
	double w_low, double w_high, std::vector<double>& W){
  std::vector<double> F1=F;
  if(w_high==w_low){
    for(int i=0;i<F.size();i++)
      if(W[i]>w_high)
	F1[i]=high;
  }
  else{
    for(int i=0;i<F.size();i++){
      double t=(W[i]-w_low)/(w_high-w_low)*high;
      F1[i]=std::max(F1[i], t);
    }
  }
  mat M;
  init(X, M, F1, high);
  reduce(M);
  print_barcode(high, minlen, M, X, F1);
  return 0;
}

/******************************************************************************/

int gengraph(int* G, cplx& X, std::vector<double>& F, int size){
  int n=G[0];
  int i=1;
  int maxdg=0;
  for(i; i<n+1; i++){
    X.push_back({0, 0, {}});
    F.push_back((double)G[i]);
    if(G[i]>maxdg)maxdg=G[i];
  }
  for(int j=0;j<n;j++)
    F[j]=1.0-F[j]/maxdg;
  for(i;i<size;i+=2){
    X.push_back({1, 2, {G[i], G[i+1]}});
    F.push_back(std::max(F[G[i]], F[G[i+1]]));
  }
  return 0;
}


int gengraph2(int* G, cplx& X, std::vector<double>& F, std::vector<double>& W, int size){
  int n=G[0];
  int i=1;
  int maxdg=0;
  int center=0;
  for(i; i<n+1; i++){
    X.push_back({0, 0, {}});
    F.push_back((double)G[i]);
    W.push_back(10);
    if(G[i]>maxdg){maxdg=G[i];
      center=i-1;
    }
  }
  for(int j=0;j<n;j++)
    F[j]=1.0-F[j]/maxdg;
  W[center]=0;
  for(i;i<size;i+=2){
    X.push_back({1, 2, {G[i], G[i+1]}});
    F.push_back(std::max(F[G[i]], F[G[i+1]]));
  }
  int flag=1;
  while(flag){flag=0;
  for(int j=n; j<X.size();j++){
    int a=X[j].bd[0];
    int b=X[j].bd[1];
    if(W[a]>W[b]+1){W[a]=W[b]+1;flag++;}
    else if(W[b]>W[a]+1){W[b]=W[a]+1; flag++;}
  }
  }
  for(int j=n;j<X.size();j++)
    W.push_back(std::max(W[X[j].bd[0]], W[X[j].bd[1]]));
  return 0;
}

int main(){
  for(int i=0;i<npts;i++){
    x[i]=((double)rand())/RAND_MAX*2;
    y[i]=((double)rand())/RAND_MAX;
  }
  clock_t t0=clock();
  FILE* input=fopen("reddit_5k", "r");
  for(int S=0;S<4999;S++){
  //Read graph
    int szgp=0;
    fscanf(input, "%d", &szgp);
    int gph[szgp-1];
    for(int i=0;i<szgp-1;i++)
      fscanf(input, "%d", gph+i);
    printf("%d", gph[0]-1);
    cplx X;
    std::vector<double> F;
    std::vector<double> W;
    gengraph(gph+1, X, F, szgp-2);
    gengraph2(gph+1, X, F, W, szgp-2);
    //PH0(1, 0.02, X, F);
    PH1(1, 0.02, X, F, 0, 5, W);
    printf("\n");
  }
  //printf("time: %lf\n", ((double)(clock()-t0))/CLOCKS_PER_SEC);
  return 0;
}
