#weighted persistent homology for theta graph

import dionysus as d
import math
import random
from multiprocessing import Pool

#Height function
def f(n):
    x=(n%400)*0.01-2
    y=((n-n%400)/400)*0.01-2
    return y

#Density function
def rho(n):
    x=(n%400)*0.01-2
    y=((n-n%400)/400)*0.01-2
    return max(math.exp(-20.0*(x*x+y*y-1)**2),math.exp(-20.0*y**4))

#Create rectangular grid

def get_grid(grid_size):
    grid=[]
    for i in range(grid_size):
        for j in range(grid_size):
            grid+=[[i*grid_size+j]]
    for i in range(grid_size-1):
        for j in range(grid_size-1):
            grid+=[[i*grid_size+j,i*grid_size+j+1],[i*grid_size+j,(i+1)*grid_size+j],[i*grid_size+j,(i+1)*grid_size+j+1],[i*grid_size+j,i*grid_size+j+1,(i+1)*grid_size+j+1],[i*grid_size+j,(i+1)*grid_size+j,(i+1)*grid_size+j+1]]
    for i in range(grid_size-1):
        grid+=[[i*grid_size+grid_size-1,(i+1)*grid_size+grid_size-1],[(grid_size-1)*grid_size+i,(grid_size-1)*grid_size+i+1]]
    return grid

grid=get_grid(400)

pw=[f(n) for n in list(range(400*400))]
pd=[1-rho(n) for n in list(range(400*400))]


mf=min(pw)
Mf=max(pw)
mrho=min(pd)
Mrho=max(pd)
rstep=(Mrho-mrho)/5


#pw=[w+random.uniform(-0.1,0.1) for w in pw]
#pd=[s+random.uniform(-0.1,0.1) for s in pd]

weight=[]
dense=[]    
for cell in grid:
    weight+=[max([pw[n] for n in cell])]
    dense+=[max([pd[n] for n in cell])]

fil_weight=[]
for v_f, v_rho in zip(weight, dense):
    if v_rho<=0.4:
        fil_weight+=[v_f]
    else:
        fil_weight+=[Mf]
f=d.Filtration()
for cell, level in zip(grid, fil_weight):
    f.append(d.Simplex(cell, level))
f.sort()
res=d.homology_persistence(f)
dgm=d.init_diagrams(res, f)
for i, dg in enumerate(dgm):
    for pt in dg:
        if pt.death-pt.birth>0.1:
            print(i, pt.birth, pt.death)
        
    
  
for l in range(2,3):
    rho_0=mrho+l*rstep
    epsilon=0.3
    epsilon0=0.4
    print("rho_0", rho_0)
    estep=epsilon*(1-epsilon0)/6
    for k in range(7):
        print("epsilon:", epsilon)
        for n in range(17):
            a=mf*(16-n)/16+Mf*n/16
            fstep=(Mf-mf)/16
            print(a)
            fil_weight=[]
            for v_f, v_rho in zip(weight, dense):
                if v_rho<=rho_0-epsilon and v_f<=a:
                    fil_weight+=[v_f]
                elif v_rho<=rho_0+epsilon and v_f>a:
                    fil_weight+=[v_f]
                elif v_rho>rho_0+epsilon:
                    fil_weight+=[Mf]
                else:
                    fil_weight+=[a]
            f=d.Filtration()
            for cell, level in zip(grid, fil_weight):
                f.append(d.Simplex(cell, level))
            f.sort()
            res=d.homology_persistence(f)
            dgm=d.init_diagrams(res,f)
            for i, dg in enumerate(dgm):
                for pt in dg:
                    if pt.birth<a-fstep*0.1 and pt.death>a+fstep*0.1 and pt.birth>=a-fstep*1.1 and pt.death-pt.birth>fstep:
                        if pt.death==float("inf"):
                            print(i, pt.birth, Mf)
                        else:
                            print(i, pt.birth, pt.death)
        epsilon-=estep

