import sys
import numpy as np
import pandas as pd
import pickle as pk
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.7):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())

with open("/tmp2/b02902030/data.pk", "rb") as fp:
    X_train = pk.load(fp) 
    X_test = pk.load(fp)
    Y_train = pk.load(fp)
    Y_test = pk.load(fp)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


lstm_out = 200
model = Sequential()
#model.add(LSTM(lstm_out, input_shape=(48, 100)))
model.add(LSTM(lstm_out, input_shape=(50, 100), dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])

batch_size = 64
best_acc=0
for i in range(200):
    print("epoch: ", i+1)
    model.fit(X_train, Y_train, epochs=1, batch_size=batch_size)
    loss,acc = model.evaluate(X_test, Y_test, batch_size=X_test.shape[0])
    if acc>best_acc:
        best_acc=acc
        model.save('/tmp2/b02902030/'+sys.argv[1]+'.h5')
    print("\nloss: ", loss, " acc: ", acc, " best_acc: ", best_acc, "\n")


