#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

int n_f;
int n_s;
struct dist{
  int id1;
  int id2;
  double dist;
};
int readfile(FILE* f, std::vector<std::vector<double> > &data){
  char buf[1024];
  while(!feof(f)){
    if(fgets(buf, 1024, f)==NULL)break;
    char* cur=strtok(buf, " ");
    std::vector<double> tmp;
    double tmpd;
    while(cur){
      sscanf(cur,"%lf",&tmpd);
      tmp.push_back(tmpd);
      cur=strtok(NULL," ");
    }
    data.push_back(tmp);
  }
  return 0;
}

bool cmp(const dist &a, const dist &b){
  return a.dist<b.dist||(a.dist==b.dist && a.id1<b.id1)||
    (a.dist==b.dist && a.id1==b.id1 && a.id2<b.id2);
}

bool cmp2(const dist &a, const dist &b){
  return a.id1<b.id1 || (a.id1==b.id1 && a.id2<b.id2);
}

int rank_features(std::vector<std::vector<double> > &d){
  double eff[n_f];
  for(int k=0;k<n_f;k++){
    std::vector<dist> ds1;
    std::vector<dist> ds2;
    for(long i=0;i<n_s;i++){
      for(long j=0;j<i;j++){
        double d1=0;
        double d2=0;
        for(int l=0;l<n_f;l++)
          if(l==k)d1+=(d[i][l]-d[j][l])*(d[i][l]-d[j][l]);
          else{
            d1+=(d[i][l]-d[j][l])*(d[i][l]-d[j][l]);
            d2+=(d[i][l]-d[j][l])*(d[i][l]-d[j][l]);
          }
        dist tmp1={j,i,d1};
        dist tmp2={j,i,d2};
        ds1.push_back(tmp1);
        ds2.push_back(tmp2);
      }
    }
    std::sort(ds1.begin(), ds1.end(), cmp);
    std::sort(ds2.begin(), ds2.end(), cmp);
    for(long i=0;i<ds1.size();i++){
      ds1[i].dist=i; 
      ds2[i].dist=i;
    }
    std::sort(ds1.begin(), ds1.end(), cmp2);
    std::sort(ds2.begin(), ds2.end(), cmp2);
    double sum=0;
    for(long i=0;i<ds1.size();i++){
      sum+=(ds1[i].dist>ds1.size()*0.8 && ds2[i].dist>ds1.size()*0.8)
       +(ds1[i].dist<ds1.size()*0.2 && ds2[i].dist<ds1.size()*0.2);
    }
    //printf("%d, %lf\n", k, sum);
    eff[k]=sum;
  }
  int r=0;
  for(int i=1;i<n_f;i++)
    if(eff[i]>eff[r])
      r=i;
  return r;
}


int delete_feature(int s, std::vector<std::vector<double> > &d){
  for(long i=0;i<n_s;i++)
    d[i].erase(d[i].begin()+s);
  return 0;
}

const long maxdim=3;
#include "ph.cpp"

typedef std::vector<double> pt;

double R;
double distr(pt p1, pt p2){
  double r=0;
  for(int i=0;i<p1.size();i++)
    r+=(p1[i]-p2[i])*(p1[i]-p2[i]);
  return r;
}


double dist(pt p1, pt p2){
  double r=0;
  for(int i=0;i<p1.size();i++)
    r+=(p1[i]-p2[i])*(p1[i]-p2[i]);
  return r/R;
}

int conn(double d, pt p1, pt p2){
  return dist(p1,p2)<d;
}

double rho(pt p){
  return 0;
}

#include "rips_cplx.cpp"

int main(int argc, char* argv[]){
  FILE* input=fopen(argv[1],"r");
  std::vector<std::vector<double> > data;
  readfile(input, data);
  n_f=data[0].size();
  n_s=data.size();
  char features[n_f];
  for(int i=0;i<n_f;i++)features[i]=0;
  while(n_f>1){
    R=0;
    for(int i=0;i<n_s;i++)
      for(int j=0;j<i;j++)
        R+=distr(data[i], data[j]);
    R/=n_s*n_s;
    std::vector<bdit> M;
    rips(1,n_s,data,M);
    reduce(1,1,M);
    int s=rank_features(data);
    int id_f=0;
    while(s>0 || features[id_f])
      if(features[id_f])id_f++;
      else{s--;id_f++;}
    printf("%d\n", id_f);
    features[id_f]=1;
    delete_feature(s, data);
    n_f--;
  }
  return 0;
}
