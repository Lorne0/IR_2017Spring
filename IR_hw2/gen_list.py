import numpy as np
import pickle as pk

with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

train_list = {}
docid = list(rel.keys())[:6693]
for d in docid:
    print(d)
    train_list[d] = []
    r = list(rel[d].values())
    for i in range(len(r)):
        for j in range(len(r)):
            if r[i]<r[j]:
                train_list[d].append((i,j))

s=0
for d in docid:
    s+=len(train_list[d])

print(s)
