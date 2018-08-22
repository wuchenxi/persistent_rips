typedef struct face{int n; long v[maxdim];long id;} fc;

double wt(fc f, pt* samples, double (*rho)(pt)){
  double max=rho(samples[f.v[0]]);
  for(long i=1;i<f.n;i++)
    if(max<(*rho)(samples[f.v[i]]))
      max=(*rho)(samples[f.v[i]]);
  return max;
}

int contains(fc f1, fc f2){
  if(f1.n!=f2.n-1)return 0;
  for(long k=0;k<f1.n;k++){
    long j;
    for(j=0;j<f2.n;j++)
      if(f2.v[j]==f1.v[k])
	break;
    if(j==f2.n)return 0;
  }
  return 1;
}

//create rips complex: d: threshold distance, pts: points, npts: num. of points
int rips(double d, long npts, pt* pts, std::vector<bdit>&M){
  //0-cells
  long vert[maxdim];
  long N=0;
  std::vector<std::vector<fc>>faces(npts);
  std::vector<std::vector<fc>>newfaces(npts);
  for(int i=0;i<npts;i++){
    fc nf;
    nf.n=1;
    nf.v[0]=i;
    nf.id=N;
    faces[i].push_back(nf);
    
    bdit ncol;
    ncol.val=0;
    ncol.w=rho(pts[i]);
    ncol.dim=0;
    M.push_back(ncol);
    N++;
  }
  for(int delta=2;delta<=maxdim;delta++){
    //move next
    for(int i=0;i<delta;i++)vert[i]=i;
    while(vert[0]<npts-delta){
      int k;
      for(k=1;k<delta;k++)
	if(vert[delta-k]<npts-k)
	  break;
      vert[delta-k]++;
      for(int j=delta-k+1;j<delta;j++)
	vert[j]=vert[delta-k]+(j-delta+k);
 
      int flag=1;
      for(k=0;k<delta;k++){
	for(int k1=k+1;k1<delta;k1++)
	  if(!conn(d, pts[vert[k]], pts[vert[k1]])){
	    flag=0;
	    break;
	  }
	if(flag==0)break;
      }
      if(!flag)continue;
      fc nf;
      nf.n=delta;
      for(k=0;k<delta;k++)nf.v[k]=vert[k];
      nf.id=N;N++;
      for(k=0;k<delta;k++)newfaces[vert[k]].push_back(nf);
      bdit ncol;
      double diam=0;
      for(k=0;k<delta;k++)
	for(int k1=0;k1<k;k1++)
	  if(dist(pts[vert[k]], pts[vert[k1]])>diam)
	    diam=dist(pts[vert[k]], pts[vert[k1]]);
      ncol.val=diam;
      ncol.w=wt(nf,pts,&rho);
      ncol.dim=delta-1;
      for(k=0;k<delta;k++)
	for(long k1=0;k1<faces[vert[k]].size();k1++)
	  if(contains(faces[vert[k]][k1], nf))
	    ncol.bdry.push_back(faces[vert[k]][k1].id);
      M.push_back(ncol);
    }
    faces=newfaces;
    for(int i=0;i<npts;i++)
      newfaces[i].clear();
  }
  return 0;
}

