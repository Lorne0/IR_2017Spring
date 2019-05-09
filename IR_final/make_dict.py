import numpy as np
import pickle as pk
import re

dic = {}
with open("glove.twitter.27B.100d.txt", "r") as fp:
    for f in fp:
        v = f.strip('\n').split(' ')
        vv = re.sub('[^a-zA-z0-9\s]','',v[0])
        if vv==v[0]:
            dic[v[0]] = np.array(list(map(float, v[1:])))

with open("dict.pk", "wb") as fp:
    pk.dump(dic, fp, protocol=pk.HIGHEST_PROTOCOL)
