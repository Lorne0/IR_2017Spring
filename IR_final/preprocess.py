import sys
import numpy as np
import pandas as pd
import pickle as pk
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split

def remove_stopword(a):
    stop=['.',',',"'",'a','an','and','am','i','you','is','was','are','were','be','he','she','it','the','of','to','for','that','those','this','these','at','so','my','your','their','did','in','on','out','what','where','when','can','could','into','me','him','her','them','if','we','us','our','which','with','how', 'they', 'have', 'had','has','his','very','been','who','also','much','as']
    #stop = ['.', ',', "'"]
    b=[]
    for s in a:
        c=[]
        for w in s:
            if w.lower() not in stop:
                c.append(w.lower())
        b.append(c)
    return b

pos = ["joy", "love"]
neg = ["anger", "sadness", "fear"]

mode = sys.argv[1]
if mode=="neg_pos" or mode=="all":
    sentiment = ["anger", "joy", "sadness", "love", "fear"]
elif mode=="neg":
    sentiment = ["anger", "sadness", "fear"]
elif mode=="pos":
    sentiment = ["joy", "love"]

sen_train = []
sen_test = []
label_train = []
label_test = []
for st in sentiment:
    with open("pkl/"+st+".pkl", "rb") as fp:
        a = pk.load(fp)
    a = remove_stopword(a)
    #a = list(map(' '.join, a))
    #a = [k.lower() for k in a]
    a = [k for k in a if len(k)<=50 and len(k)>=5]
    ts = int(len(a)*0.9)
    sen_train += a[:ts]
    sen_test += a[ts:]

    if mode=="neg_pos":
        if st in pos:
            label_train += (["positive"]*len(a))[:ts]
            label_test += (["positive"]*len(a))[ts:]
        elif st in neg:
            label_train += (["negative"]*len(a))[:ts]
            label_test += (["negative"]*len(a))[ts:]
    else:
        label_train += ([st]*len(a))[:ts]
        label_test += ([st]*len(a))[ts:]
    

label_train = np.array(label_train)
label_test = np.array(label_test)

'''
ln = list(map(len, sen_train))
lt = list(map(len, sen_test))
l = ln+lt
l = sorted(l, reverse=True)
print(l[:100])
'''
#timestep=np.max(l)
timestep=50
data_size = len(sen_train)+len(sen_test)
print("data_size: ", data_size)
print("train: ", len(sen_train))
print("test: ", len(sen_test))
print("timestep: ", timestep)

with open("/tmp2/b02902030/dict.pk", "rb") as fp:
    dic = pk.load(fp)

Zero = np.zeros(100)
X_train = np.zeros((len(sen_train), timestep, 100))
for i in range(len(sen_train)):
    for j in range(len(sen_train[i])):
        s = sen_train[i][j]
        if s in dic:
            X_train[i][j] = dic[s]
        else:
            X_train[i][j] = Zero
X_test = np.zeros((len(sen_test), timestep, 100))
for i in range(len(sen_test)):
    for j in range(len(sen_test[i])):
        s = sen_test[i][j]
        if s in dic:
            X_test[i][j] = dic[s]
        else:
            X_test[i][j] = Zero

Y_train = pd.get_dummies(label_train).values
Y_test = pd.get_dummies(label_test).values
#print(Y[0])
#print(label[0])


r = list(range(len(sen_train)))
np.random.shuffle(r)
X_train = X_train[r]
Y_train = Y_train[r]
r = list(range(len(sen_test)))
np.random.shuffle(r)
X_test = X_test[r]
Y_test = Y_test[r]


with open("/tmp2/b02902030/data.pk", "wb") as fp:
    pk.dump(X_train, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_test, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_test, fp, protocol=pk.HIGHEST_PROTOCOL)


'''
sentences = np.array(sentences)
label = np.array(label)

tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X)

Y = pd.get_dummies(label).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 1229)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

with open("data.pk", "wb") as fp:
    pk.dump(X_train, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(X_test, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fp, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_test, fp, protocol=pk.HIGHEST_PROTOCOL)
'''
