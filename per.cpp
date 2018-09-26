//Experiment 1: Theta graph, rips complex

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

const int npt=250;

/***************Persistent Homology*************************/
const int maxbd=3;
const int maxdim=1;
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
  for(int i=0;i<N;i++){//if(i%10000==0)printf("col %d\n", i);
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

int print_barcode(double high, double minlen, mat& M, cplx& X, std::vector<double>& F){
  int N=M.size();
  std::vector<char> L(N, 0);
  for(int i=0;i<N;i++){
    if(M[i].bd.size()){
      int low=M[i].bd.back();
      L[low]=1;
      if(X[M[low].id].dim<=maxdim){
	double birth=F[M[low].id];
	double death=F[M[i].id];
	if(death-birth>=minlen){
	  printf("%d %lf %lf\n", X[M[low].id].dim, birth, death); 
	}
      }
    }
  }
  for(int i=0;i<N;i++){
    if(L[i]==0 && M[i].bd.size()==0 && X[M[i].id].dim<=maxdim){
      double birth=F[M[i].id];
      double death=high;
      if(death-birth>=minlen){
	printf("%d %lf %lf\n", X[M[i].id].dim, birth, death);
      }
    }
  }
  return 0;
}


int print_barcode(double birth_low, double birth_high, double high, double minlen, mat& M, cplx& X, std::vector<double>& F){
  int N=M.size();
  std::vector<char> L(N, 0);
  for(int i=0;i<N;i++){
    if(M[i].bd.size()){
      int low=M[i].bd.back();
      L[low]=1;
      if(X[M[low].id].dim<=maxdim){
	double birth=F[M[low].id];
	double death=F[M[i].id];
	if(death-birth>=minlen && birth>=birth_low && birth<birth_high){
	  printf("%d %lf %lf\n", X[M[low].id].dim, birth, death); 
	}
      }
    }
  }
  for(int i=0;i<N;i++){
    if(L[i]==0 && M[i].bd.size()==0 && X[M[i].id].dim<=maxdim){
      double birth=F[M[i].id];
      double death=high;
      if(death-birth>=minlen && birth>=birth_low && birth<birth_high){
	printf("%d %lf %lf\n", X[M[i].id].dim, birth, death);
      }
    }
  }
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

int PH2(double high, double minlen, cplx X, std::vector<double>& F,
	double w_low, double w_high, double epsilon, std::vector<double>& W){
  std::vector<double> A;
  for(double a=0;a<high;a+=minlen)
    A.push_back(a);
  for(int j=1;j<A.size();j++){
    std::vector<double> F1=F;
    if(w_high==w_low){
      for(int i=0;i<F.size();i++)
	if(W[i]>w_high+epsilon)
	  F1[i]=high;
        else if(W[i]>w_high && F[i]<A[j])
	  F1[i]=A[j];
    }
    else{
      for(int i=0;i<F.size();i++){
	double t=(W[i]-w_low)/(w_high-w_low)*high;
	double t1=std::max(F1[i], t);
	if(t1<A[j])
	  F1[i]=t1;
	else{
	  t=(W[i]-epsilon-w_low)/(w_high-w_low)*high;
	  t1=std::max(t, F1[i]);
	  if(t1<A[j])
	    F1[i]=A[j];
	  else
	    F1[i]=t1;
	}
      }      
    }
    mat M;
    init(X, M, F1, high);
    reduce(M);
    print_barcode(A[j-1], A[j], high, minlen, M, X, F1);
  }
  return 0;
}

/******************************************************************************/
//2d rips complex
bool cmp(face f, face cur){
  return f.bd[0]<cur.bd[0]||(f.bd[0]==cur.bd[0]&&f.bd[1]<cur.bd[1]);
}

int rips(cplx& X, std::vector<double>& F, std::vector<double>& W, long N, double max_dist, double* w, double (*dist)(int, int)){
  for(int i=0;i<N;i++){
    F.push_back(0);
    X.push_back({0, 0, {}});
    W.push_back(w[i]);
  }
  for(int i=0;i<N;i++){
    for(int j=i+1;j<N;j++){
      double d;
      if((d=dist(i,j))<=max_dist){
	X.push_back({1, 2, {i, j}});
	F.push_back(d);
	W.push_back(std::max(w[i], w[j]));
      }
    }
  }
  int nedges=X.size();
  for(int i=0;i<N;i++){
    for(int j=i+1;j<N;j++){
      for(int k=j+1;k<N;k++){		
	face bdf={1, 2, {i, j}};
	int r1=std::lower_bound(X.begin()+N, X.begin()+nedges, bdf, cmp)-X.begin();
	if(r1==nedges||cmp(bdf, X[r1]))continue;
	double fil=F[r1];
	bdf={1,2, {i, k}};
	int r2=std::lower_bound(X.begin()+N, X.begin()+nedges, bdf, cmp)-X.begin();
	if(r2==nedges||cmp(bdf, X[r2]))continue;
	double fil1=F[r2];fil=std::max(fil, fil1);
	bdf={1, 2, {j, k}};
	int r3=std::lower_bound(X.begin()+N, X.begin()+nedges, bdf , cmp)-X.begin();
	if(r3==nedges||cmp(bdf, X[r3]))continue;
	fil1=F[r3];fil=std::max(fil, fil1);
	X.push_back({2, 3, {r1, r2, r3}});
	F.push_back(fil);
	W.push_back(std::max(W[r1], w[k]));
      }
    }
  }
  return 0;
}

/******************************************************************************/

double x[npt];
double y[npt];

double dist(int i, int j){
  return sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]));
}

double get_dtm(int i, int n, double m){
  std::vector<double>dists(n);
  for(int j=0;j<n;j++)dists[j]=dist(i, j);
  std::sort(dists.begin(), dists.end());
  double sum=0;
  for(int j=1; j<m*n+1;j++)sum+=dists[j]*dists[j];
  return sqrt(sum/((int)(m*n)));
}

double norm(){
  double r=0;
  for(int i=0; i<12; i++)
    r+=((double)rand())/RAND_MAX-0.5;
  return r;
}

int main(){
  clock_t t0=clock();
  for(int S=0;S<5;S++){
  //Theta graph
  for(int i=0;i<npt;i++){
    if(i%2){
      double s=((double)rand())*M_PI*2.0/RAND_MAX;
      x[i]=sin(s)+norm()*0.1;
      y[i]=cos(s)+norm()*0.1;
    }
    else{
      double s=((double)rand())/RAND_MAX*2.0-1;
      x[i]=s+norm()*0.1;
      y[i]=s+norm()*0.1;
    }
  }
  cplx X;
  double dtm[npt];
  for(int i=0;i<npt;i++)dtm[i]=get_dtm(i, npt, 0.05);
  double dtm_high=dtm[0];
  double dtm_low=dtm[0];
  for(int i=0;i<npt;i++){
    if(dtm[i]>dtm_high)dtm_high=dtm[i];
    if(dtm[i]<dtm_low)dtm_low=dtm[i];
  }

  /*********************************************************/
  std::vector<double> F;
  std::vector<double> W;
  rips(X, F, W, npt, 1, dtm, dist);
  printf("Number of faces %ld\n", X.size());
  printf("dtm %lf %lf\n", dtm_high, dtm_low);
  printf("no thresholding\n");
  PH0(1, 0.05, X, F);
  printf("hard thresholding\n");
  PH1(1, 0.05, X, F, 0.3, 0.3, W);
  printf("soft thresholding 1\n");
  PH1(1, 0.05, X, F, 0.2, 0.4, W);
  printf("soft thresholding 2\n");
  PH2(1, 0.05, X, F, 0.29, 0.29, 0.02, W);
  printf("soft thresholding 3\n");
  PH2(1, 0.05, X, F, 0.2, 0.4, 0.02, W);
}
  printf("time: %lf\n", ((double)(clock()-t0))/CLOCKS_PER_SEC);
  return 0;
}
