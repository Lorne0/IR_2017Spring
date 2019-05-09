import numpy as np
import pickle as pk

with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
    feature = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

n=136
list_rel_matrix = {}
qid = list(rel.keys())
for q in qid:
    #print(q)
    list_rel_matrix[q] = {}
    #list_matrix[q][0] = rA^T*A
    #list_matrix[q][1] = rA^T*r
    l = len(feature[q])
    A = np.zeros((l,n))
    for i in range(l):
        A[i] = feature[q][i+1]
    AT = np.transpose(A)
    r = list(rel[q].values())
    r2 = (np.array(r)+1)**2
    rAT = np.multiply(AT, r2)
    list_rel_matrix[q][0] = np.dot(rAT, A)
    list_rel_matrix[q][1] = np.dot(rAT, r)

with open("/tmp2/b02902030/IR_hw2/list_rel_matrix.pk", "wb") as fp:
    pk.dump(list_rel_matrix, fp, protocol=pk.HIGHEST_PROTOCOL)
