//weighted rips complex of theta graph

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

const long maxdim=3;
#include "ph.cpp"

typedef struct point{double x; double y;} pt;


double dist(pt p1, pt p2){
  double dx=p1.x-p2.x;
  double dy=p1.y-p2.y;
  return sqrt(dx*dx+dy*dy);
}

int conn(double dist_threshold, pt p1, pt p2){
  return dist(p1,p2)<dist_threshold;
}


double rho(pt p){double d=p.x*p.x+p.y*p.y-1;
  if(fabs(p.y)<d)d=fabs(p.y);
  return d*d;}

#include "rips_cplx.cpp"

int getdata(long n, long n2, pt* samples){
  for(long i=0;i<n;i++){
    double e1=(rand()%1000-500.0)/10000;
    double e2=(rand()%1000-500.0)/10000;
    double s=i;
    samples[i].x=cos(s)+e1;
    if(i<n2)
      samples[i].y=sin(s)+e2;
    else
      samples[i].y=e2;
  }
  return 0;
}


int main(){
  std::vector<bdit> M;
  pt samples[1000];
  getdata(1000, 600, samples);
  rips(0.2,1000,samples,M);
  reduce(1,1.4,M);
  return 0;
}
