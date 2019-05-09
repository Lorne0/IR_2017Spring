import numpy as np
import pickle as pk
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

model = load_model('model_adam.h5')

with open("/tmp2/b02902030/IR_hw2/test_feature.pk", "rb") as fp:
    test_feature = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/X_test.pk", "rb") as fp:
    X_test = pk.load(fp)

y = model.predict(X_test)
y = np.array([y[i][0] for i in range(65059)])

test_qid = list(test_feature.keys())
test_id = {}
test_rel = {}

cnt=0
for q in test_qid:
    test_id[q] = {}
    test_rel[q] = {}
    for d in test_feature[q]:
        test_id[q][d] = cnt
        cnt+=1
print(cnt)

for q in test_qid:
    for d in test_feature[q]:
        test_rel[q][d] = y[test_id[q][d]]

with open("output_NN.csv", "w") as fp:
    fp.write("QueryId,DocumentId\n")
    for q in test_qid:
        wx = np.array(list(test_rel[q].values()))
        swx = np.argsort(wx)
        swx = swx[::-1]
        swx = swx[:10]
        for i in swx:
            fp.write(str(q)+","+str(i)+"\n")
