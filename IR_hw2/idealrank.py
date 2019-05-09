import numpy as np
import pickle as pk

with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

qid = list(rel.keys())

IDCG = {}
for q in qid:
    r = list(rel[q].values())
    sr = np.sort(r)
    sr = sr[::-1]
    #print(sr[:10])
    s = 0
    if len(sr)<10:
        for i in range(1,len(sr)+1):
            s += ((np.power(2,sr[i-1])-1)/np.log2(i+1))
    else:
        for i in range(1,11):
            s += ((np.power(2,sr[i-1])-1)/np.log2(i+1))
    IDCG[q] = s


with open("IDCG.pk", "wb") as fp:
    pk.dump(IDCG, fp, protocol=pk.HIGHEST_PROTOCOL)

