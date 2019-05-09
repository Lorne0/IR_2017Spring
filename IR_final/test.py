import numpy as np
import pickle as pk
from keras.models import load_model
import re
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())

def remove_stopword(s):
    b=[]
    stop=['.',',',"'",'a','an','and','am','i','you','is','was','are','were','be','he','she','it','the','of','to','for','that','those','this','these','at','so','my','your','their','did','in','on','out','what','where','when','can','could','into','me','him','her','them','if','we','us','our','which','with','how', 'they', 'have', 'had','has','his','very','been','who','also','but','much','as']
    for w in s:
        if w.lower() not in stop:
            b.append(w.lower())
    return b

with open("./dict.pk", "rb") as fp:
    dic = pk.load(fp)
neg_pos = load_model('./neg_pos.h5')
neg = load_model('./neg.h5')
pos = load_model('./pos.h5')

while(1):
    x = np.zeros((1, 50, 100))
    sen = input("Enter you tweet: ")
    s = re.sub('[^a-zA-z0-9\s]','',sen)
    s = s.split(' ')
    s = remove_stopword(s)
    for i in range(len(s)):
        if s[i] in dic:
            x[0][i] = dic[s[i]]
        else:
            x[0][i] = np.zeros(100)
    
    s1 = neg_pos.predict(x)[0]
    s2 = neg.predict(x)[0] if s1[0]>s1[1] else pos.predict(x)[0]
    
    senti = ["neg", "pos", "neg", "pos", "neg"]
    sneg = ["anger", "sadness", "fear"]
    spos = ["joy", "love"]
    if s1[0]>s1[1]:
        print(s1, s2, senti[np.argmax(s1)], sneg[np.argmax(s2)])
    else:
        print(s1, s2, senti[np.argmax(s1)], spos[np.argmax(s2)])
    



