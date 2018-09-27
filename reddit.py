import scipy as SP
import csv
from random import shuffle
from sklearn.svm import SVC
clf=SVC(gamma="auto")

data=SP.array(list(csv.reader(open("out_reddit", "r"), delimiter=' '))).astype(float)
sz=data.shape
print(sz)
idx=list(range(sz[0]))
shuffle(idx)
train_idx=idx[:3500]
test_idx=idx[3500:]
train_X=data[train_idx][:,1:]
train_Y=data[train_idx][:,0].astype(int)
test_X=data[test_idx][:,1:]
test_Y=data[test_idx][:,0].astype(int)
#xg_train=xgb.DMatrix(train_X, label=train_Y)
#xg_test=xgb.DMatrix(test_X)
#param={'objective' : 'multi:softmax', 'eta' : 0.1, 'num_class': 5, 'silent': 1}
#bst=xgb.train(param, xg_train, 50)
#pred_Y=bst.predict(xg_test)
clf.fit(train_X, train_Y)
pred_Y=clf.predict(test_X)
c=0
a=0
for i in range(pred_Y.shape[0]):
    if pred_Y[i]==test_Y[i]:
        c+=1
    a+=1
print(c*1.0/a)
