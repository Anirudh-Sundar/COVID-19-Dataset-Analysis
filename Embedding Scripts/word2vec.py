# -*- coding: utf-8 -*-


import logging
import pandas as pd
import numpy as np
import gensim
import nltk
import re
from bs4 import BeautifulSoup



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')





 
#from itertools import islice


wv = gensim.models.KeyedVectors.load_word2vec_format('D:/data/word_vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
wv.init_sims(replace=True) 
from sklearn.externals import joblib 
joblib.dump(wv, './data/w2v.pkl') 

from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import gmean

def word_averaging(wv, words):
    all_words, mean = set(), []
    cou=0
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    #mean = gensim.matutils.unitvec(gmean(np.array(mean))).astype(np.float32)
    #mean = gensim.matutils.unitvec(np.percentile(np.array(mean),25,axis=0)).astype(np.float32)
    #mean = gensim.matutils.unitvec(skew(np.array(mean))).astype(np.float32)
   #mean = gensim.matutils.unitvec(np.percentile(np.array(mean),50,axis=0)).astype(np.float32)
    return mean


def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            tokens.append(word)
    return tokens



fname='abstract.csv'
df = pd.read_csv(fname,encoding='latin-1')
comtdata=df
test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r[0]), axis=1).values
X_comtdata_average1 = word_averaging_list(wv,test_tokenized)
y=df['label']
y=np.array(y)

x_sum=np.sum(X_comtdata_average1,axis=1)
ino=np.where(x_sum!=0)
x=X_comtdata_average1[ino[0],:]
y=y[ino[0]]


X_comtdata_averagen=np.concatenate((x, y.reshape(-1,1)), axis=1)
fname='word2vec.csv'
np.savetxt(fname,X_comtdata_averagen, delimiter=',', fmt='%f')



