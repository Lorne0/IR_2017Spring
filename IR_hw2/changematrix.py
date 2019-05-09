import numpy as np
import pickle as pk
import random

with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
    feature = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

def sigmoid(x):
    return 1/(1+np.exp(-x))

X_query = {}
y_query = {}
y_query_exp = {}
y_query_sigmoid = {}

qid = list(rel.keys())
for q in qid:
    y = np.array(list(rel[q].values()))
    y_query[q] = (y+1) / np.sum(y+1)

    z = np.sum(np.exp(y))
    y_query_exp[q] = np.exp(y)/z

    y_query_sigmoid[q] = sigmoid(y) / np.sum(sigmoid(y))

    X_query[q] = np.zeros((len(rel[q]),136))
    c=0
    for d in rel[q]:
        X_query[q][c] = feature[q][d]
        c+=1

with open("/tmp2/b02902030/IR_hw2/X_query.pk", "wb") as fp:
    pk.dump(X_query, fp, protocol=pk.HIGHEST_PROTOCOL)
with open("/tmp2/b02902030/IR_hw2/y_query.pk", "wb") as fp:
    pk.dump(y_query, fp, protocol=pk.HIGHEST_PROTOCOL)
with open("/tmp2/b02902030/IR_hw2/y_query_exp.pk", "wb") as fp:
    pk.dump(y_query_exp, fp, protocol=pk.HIGHEST_PROTOCOL)
with open("/tmp2/b02902030/IR_hw2/y_query_sigmoid.pk", "wb") as fp:
    pk.dump(y_query_sigmoid, fp, protocol=pk.HIGHEST_PROTOCOL)
