import numpy as np
import pickle as pk

train_file = "/tmp2/b02902030/IR_hw2/train.txt"
test_file = "/tmp2/b02902030/IR_hw2/test.txt"

train_rel = {}
train_feature = {}
test_feature = {}

all_vec = np.zeros((945306, 136))
cnt=0
with open(train_file) as fp:
    for f in fp:
        f = f.strip("\n")
        a = f.split()
        qid = int(a[1].split(":")[1])
        did = int(a[2].split(":")[1])
        if qid not in train_rel:
            train_rel[qid] = {}
        train_rel[qid][did] = int(a[0])
        a = a[3:]
        v = np.zeros(136)
        for i in range(len(a)):
            v[i] = float(a[i].split(":")[1])
        if qid not in train_feature:
            train_feature[qid] = {}
        train_feature[qid][did] = v
        all_vec[cnt]=v
        cnt+=1

minx = abs(np.amin(all_vec,axis=0))
maxx = abs(np.amax(all_vec,axis=0))
for q in train_feature:
    for d in train_feature[q]:
        train_feature[q][d] = (train_feature[q][d]+minx) / (maxx+minx)

with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "wb") as fp:
    pk.dump(train_rel, fp, protocol=pk.HIGHEST_PROTOCOL)
with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "wb") as fp:
    pk.dump(train_feature, fp, protocol=pk.HIGHEST_PROTOCOL)
        
print("Train done.")

#all_vec = np.zeros((65059, 136))
#cnt=0
with open(test_file) as fp:
    for f in fp:
        f = f.strip("\n")
        a = f.split()
        qid = int(a[1].split(":")[1])
        did = int(a[2].split(":")[1])
        a = a[3:]
        v = np.zeros(136)
        for i in range(len(a)):
            v[i] = float(a[i].split(":")[1])
        if qid not in test_feature:
            test_feature[qid] = {}
        test_feature[qid][did] = v
        #all_vec[cnt]=v
        #cnt+=1

#minx = abs(np.amin(all_vec,axis=0))
#maxx = abs(np.amax(all_vec,axis=0))
for q in test_feature:
    for d in test_feature[q]:
        test_feature[q][d] = (test_feature[q][d]+minx) / (maxx+minx)

with open("/tmp2/b02902030/IR_hw2/test_feature.pk", "wb") as fp:
    pk.dump(test_feature, fp, protocol=pk.HIGHEST_PROTOCOL)


print("Test done.")
