import numpy as np
import pickle as pk

with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
    feature = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

n=136
list_matrix = {}
qid = list(rel.keys())
for q in qid:
    print(q)
    list_matrix[q] = {}
    #list_matrix[q][0] = A^T*A
    #list_matrix[q][1] = A^T*r
    l = len(feature[q])
    A = np.zeros((l,n))
    for i in range(l):
        A[i] = feature[q][i+1]
    AT = np.transpose(A)
    r = list(rel[q].values())
    list_matrix[q][0] = np.dot(AT, A)
    list_matrix[q][1] = np.dot(AT, r)

with open("/tmp2/b02902030/IR_hw2/list_matrix.pk", "wb") as fp:
    pk.dump(list_matrix, fp, protocol=pk.HIGHEST_PROTOCOL)
