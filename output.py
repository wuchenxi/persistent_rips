import pickle

f=open("reddit_multi_5K.graph", "rb")
data=pickle.load(f, encoding='latin1')
for i in range(len(data['labels'])):
    out=[0, int(data['labels'][i])]
    gp=list(data['graph'][i].values())
    n=len(gp)
    out+=[n]
    for j in range(n):
        out+=[len(gp[j]['neighbors'])]
    for j in range(n):
        for k in gp[j]['neighbors']:
            if k>j:
                out+=[j, k]
    out[0]=len(out)
    print(" ".join(map(str, out)))
    
    
