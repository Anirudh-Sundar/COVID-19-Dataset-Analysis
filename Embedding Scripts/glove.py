# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
import gensim
import nltk
import re
from bs4 import BeautifulSoup
import csv

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')




wv = pd.read_table('D:/data/word_vec/glove.6B/glove.6B.300d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
from sklearn.externals import joblib 
joblib.dump(wv, './data/glovef.pkl') 

#from itertools import islice





def vec(w):
  return wv.loc[w].as_matrix()


#from scipy.stats import kurtosis
#from scipy.stats import skew
#from scipy.stats import gmean


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens



def wordprint(words,wv):
    all_words, mean = set(), []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.index:
           a=np.array(wv.loc[word])
           mean.append(a)
    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(300,)   
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean
     


def  word_averaging_listn(wv, text_list):
    return np.vstack([wordprint(post,wv) for post in text_list ])



fname='abstract.csv'
df = pd.read_csv(fname,encoding='latin-1')
comtdata=df
test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r[0]), axis=1).values
X_comtdata_averagen=word_averaging_listn(wv, test_tokenized)


x_sum=np.sum(X_comtdata_averagen,axis=1)
ino=np.where(x_sum!=0)
x=X_comtdata_averagen[ino[0],:]

fname='glove.csv'
np.savetxt(fname,X_comtdata_averagen, delimiter=',', fmt='%f')


