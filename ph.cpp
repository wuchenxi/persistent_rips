//column reduction with a twist

typedef struct col{int dim; std::vector<long> bdry; double val; double w;} bdit;
const double minlen=0.05;

int merge(std::vector<long>& cur, const std::vector<long>& pre){
  std::vector<long> tmp=cur;
  cur.clear();
  const long l1=tmp.size();
  const long l2=pre.size();
  long a=0; long b=0;
  while(a<l1||b<l2){
    if(a==l1){cur.push_back(pre[b]); b++;}
    else if(b==l2){cur.push_back(tmp[a]); a++;}
    else if(tmp[a]<pre[b]){cur.push_back(tmp[a]); a++;}
    else if(tmp[a]>pre[b]){cur.push_back(pre[b]); b++;}
    else {a++; b++;}    
  }
  return 0;
}


long rearrange(std::vector<bdit>& M){
  std::vector<bdit> res;
  std::vector<long> ref(M.size());
  std::vector<long> rref(M.size());
  for(long i=0;i<M.size();i++)ref[i]=i;
  std::sort(ref.begin(), ref.end(), [&M](long t1, long t2){
      return (M[t1].val<M[t2].val||
	      (M[t1].val==M[t2].val && t1<t2));});
  for(long i=0;i<M.size();i++)rref[ref[i]]=i;
  for(long i=0;i<M.size();i++){
    bdit cur=M[ref[i]];
    for(int j=0;j<cur.bdry.size();j++)
      cur.bdry[j]=rref[cur.bdry[j]];
    std::sort(cur.bdry.begin(),cur.bdry.end());
    res.push_back(cur);
  }
  M=res;
  return 0;
}


long rearrange(double w, double a, std::vector<long>& pos, std::vector<long>& npos, std::vector<bdit>& M){
  std::vector<bdit> res;
  for(long i=0;i<M.size();i++)
    if(M[i].val<=a && M[i].w>w)M[i].val=a;
  std::vector<long> ref(M.size());
  std::vector<long> rref(M.size());
  for(long i=0;i<M.size();i++)ref[i]=i;
  std::sort(ref.begin(), ref.end(), [&M, w](long t1, long t2){
      return (M[t1].val<M[t2].val||
	      (M[t1].val==M[t2].val && M[t1].w<=w && M[t2].w>w)||
	      (M[t1].val==M[t2].val && M[t1].w<=w && M[t2].w<=w && t1<t2)||
	      (M[t1].val==M[t2].val && M[t1].w>w && M[t2].w>w && t1<t2));});
  for(long i=0;i<M.size();i++)rref[ref[i]]=i;
  for(long i=0;i<M.size();i++){
    bdit cur=M[ref[i]];
    for(int j=0;j<cur.bdry.size();j++)
      cur.bdry[j]=rref[cur.bdry[j]];
    std::sort(cur.bdry.begin(),cur.bdry.end());
    res.push_back(cur);
  }
  M=res;
  for(long i=0;i<npos.size();i++)npos[i]=rref[pos[i]];
  return 0;
}

int reduce(double low, double high, std::vector<bdit>& M0){
  rearrange(M0);
  std::vector<long> L(M0.size());
  std::vector<bdit> M=M0;
  std::vector<long> generators;

  //First phase
  for(long delta=maxdim-1; delta>0; delta--){
    for(long i=0;i<M.size();i++){
      if(M[i].w>low||M[i].dim!=delta)
	continue;
      long flag=1;
      std::vector<long>d=M[i].bdry;
      while(d.size()>0 && L[d[d.size()-1]]>0){
	long j=L[d[d.size()-1]];
	merge(d, M[j].bdry);
      }
      if(d.size()>0){
	long m=d[d.size()-1];
	L[m]=i;
	if(M[m].dim<maxdim-1 && M[m].val<M[i].val-minlen){
	  generators.push_back(m);
	  //printf(":%ld, %lf, %lf\n", m, M[m].val, M[i].val);
	}
	M[m].bdry.clear();
      }
      M[i].bdry=d;
    }
  }
  for(long i=0;i<M.size();i++){
    if(M[i].w<=low && L[i]==0 && M[i].bdry.size()==0 && M[i].dim<maxdim-1){
      generators.push_back(i);
      //printf(":%ld, %d, %lf\n", i, M[i].dim, M[i].val);
    }
  }
  
  std::sort(generators.begin(),generators.end());

  //second phase
  double A=M0[generators[0]].val-minlen;
  long checking=0;
  for(long step=0; step<generators.size(); step++){
    if(M0[generators[step]].val<=A)continue;
    A=M0[generators[step]].val+minlen;
    checking=step;
    //printf("A=%lf\n",A);

    M=M0;
    std::vector<long> newgen(generators.size());
    rearrange(low, A, generators, newgen, M);
    for(long i=0;i<M.size();i++)L[i]=0;
    for(long delta=maxdim-1; delta>0; delta--){
      for(long i=0;i<M.size();i++){
	if(M[i].dim!=delta||M[i].w>high)
	  continue;
	long flag=1;
	std::vector<long>d=M[i].bdry;
	while(d.size()>0&&L[d[d.size()-1]]>0){
	  long j=L[d[d.size()-1]];
	  merge(d, M[j].bdry);
	}
	if(d.size()>0){
	  long m=d[d.size()-1];
	  L[m]=i;
	  M[m].bdry.clear();
	}
	M[i].bdry=d;
      }
    }
    for(long i=checking;i<newgen.size();i++)
      if(M[newgen[i]].val<A){
	long startpos=newgen[i];
	if(L[startpos]==0)printf("%d, %lf, inf\n",M[startpos].dim, M[startpos].val);
	else if(M[L[startpos]].val>M[startpos].val+minlen) printf("%d, %lf, %lf\n", M[startpos].dim, M[startpos].val, M[L[startpos]].val); 
    }
  }
  return 0;
}
