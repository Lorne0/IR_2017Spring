import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from google_scraper import scraper
import time, random
import re
from keras.models import load_model


def remove_stopword(s):
    b=[]
    stop=['.',',',"'",'a','an','and','am','i','you','is','was','are','were','be','he','she','it','the','of','to','for','that','those','this','these','at','so','my','your','their','did','in','on','out','what','where','when','can','could','into','me','him','her','them','if','we','us','our','which','with','how', 'they', 'have', 'had','has','his','very','been','who','also','but','much','as']
    for w in s:
        if w.lower() not in stop:
            b.append(w.lower())
    return b

with open("dict.pk", "rb") as fp:
    dic = pk.load(fp)

with open('tfidf.pk', 'rb') as fp:
    vec = pk.load(fp)
    voc = pk.load(fp)

neg_pos = load_model('neg_pos.h5')
neg = load_model('neg.h5')
pos = load_model('pos.h5')


cnt=0
while(1):
    cnt+=1
    corpus = input("\nWhat's happening?\n")
    corpus = [corpus]
    X_test = vec.transform(corpus)
    X_test = np.array(X_test.todense())

    x = np.zeros((1, 50, 100))
    s = re.sub('[^a-zA-z0-9\s]','',corpus[0])
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
    sentiment = ''
    if s1[0]>s1[1]:
        sentiment = sneg[np.argmax(s2)]
    else:
        sentiment = spos[np.argmax(s2)]
    
    s = np.argsort(X_test[0])[::-1]
    s = s[:3]
    scraper(sentiment+' '+' '.join(voc[s]), 5, 'image_test/', cnt)
    #time.sleep(3)


