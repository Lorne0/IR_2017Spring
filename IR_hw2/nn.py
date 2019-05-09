import numpy as np
import pickle as pk
#from sklearn.ensemble import ExtraTreesClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.3):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())

def valid(v, qid, feature, rel):
    with open("IDCG.pk", "rb") as fp:
        IDCG = pk.load(fp)
    nDCG = {}
    for q in qid:
        l = len(feature[q])
        wx = np.array(list(v[q].values()))
        swx = np.argsort(wx)
        swx = swx[::-1]
        swx+=1
        r = np.zeros(l)
        for i in range(l):
            r[i] = rel[q][swx[i]]
        s = 0
        if len(r)<10:
            for i in range(1, len(r)+1):
                s += (np.power(2, r[i-1])-1) / np.log2(i+1)
        else:
            for i in range(1, 11):
                s += (np.power(2, r[i-1])-1) / np.log2(i+1)
        if IDCG[q]==0:
            nDCG[q] = 0
        else:
            nDCG[q] = s/IDCG[q]
    return sum(list(nDCG.values()))/len(nDCG)

def custom_loss(y_true, y_pred):
    y_pred = K.exp(y_pred) / K.sum(K.exp(y_pred))
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

'''
with open("/tmp2/b02902030/IR_hw2/X_train.pk", "rb") as fp:
    X_train = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/y_train.pk", "rb") as fp:
    y_train = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/X_all.pk", "rb") as fp:
    X_train = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/y_all.pk", "rb") as fp:
    y_train = pk.load(fp)
'''
with open("/tmp2/b02902030/IR_hw2/X_query.pk", "rb") as fp:
    X_query = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/y_query.pk", "rb") as fp:
    y_query = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
    feature = pk.load(fp)
with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
    rel = pk.load(fp)

train_qid = list(rel.keys())[:6693]
valid_qid = list(rel.keys())[6693:]

X_valid = np.zeros((86323, 136))
valid_id = {}
valid_rel = {}
cnt=0
for q in valid_qid:
    valid_id[q] = {}
    valid_rel[q] = {}
    for d in rel[q]:
        X_valid[cnt] = feature[q][d]
        valid_id[q][d] = cnt
        cnt+=1
print(cnt)


print("Load data done.")

#forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)
#forest.fit(X_train, y_train)
#y = forest.predict(X_valid)
#print(y[:100])


model = Sequential()
#model.add(Dense(400, input_dim=136, activation='relu'))
model.add(Dense(400, input_dim=136, activation='linear'))
#model.add(LeakyReLU(alpha=.001))
model.add(PReLU())
#model.add(Dropout(0.5))

#model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='linear'))
#model.add(LeakyReLU(alpha=.001))
model.add(PReLU())
#model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))
#model.add(LeakyReLU(alpha=.001))
model.add(PReLU())

rms = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss = custom_loss, optimizer=rms, metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

'''
model = Sequential()
model.add(Dense(400, input_dim=136, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss = custom_loss, optimizer='rmsprop', metrics=['accuracy'])
'''


pndcg = 0

epoch = 5000
for e in range(epoch):
    #y_train = np.array(list(map(int, y_train)))
    for q in train_qid:
        l = len(rel[q])
        model.fit(X_query[q], y_query[q], epochs=1, batch_size=l, verbose=0)

    y = model.predict(X_valid)
    y = np.array([y[i][0] for i in range(86323)])

    for q in valid_qid:
        for d in rel[q]:
            valid_rel[q][d] = y[valid_id[q][d]]

    ndcg = valid(valid_rel, valid_qid, feature, rel)
    #print("epoch:", e+1, "ndcg:", ndcg, "best ndcg:", pndcg)

    if ndcg>pndcg:
        pndcg = ndcg
        model.save('model_list2.h5')
    
    print("epoch:", e+1, "ndcg:", ndcg, "best ndcg:", pndcg)


'''
###################### TEST ###########################

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
'''
