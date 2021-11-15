# -*- coding: utf-8 -*-


import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import gensim 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 


sample = open("new.txt", "r",encoding="utf8") 
s = sample.read() 
f = s.replace("\n", " ") 

data = [] 
for i in sent_tokenize(f): 
	temp = [] 
	
	# tokenize the sentence into words 
	for j in word_tokenize(i): 
		temp.append(j.lower()) 

	data.append(temp) 
model1 = gensim.models.Word2Vec(data, min_count = 1, 
							size = 100, window = 5) 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
											window = 5, sg = 1) 



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

from sklearn.externals import joblib 
joblib.dump(model1, './data/model1.pkl') 
joblib.dump(model2, './data/model2.pkl') 





def wordprint(words,model1):
    all_words, mean = set(), []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in model1.wv.vocab:
            a=np.array(model1[word])
            mean.append(a)
    if not mean:
        # FIXME: remove these examples in pre-processing
        return np.zeros(100,)   
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_listn(wv, text_list):
    return np.vstack([wordprint(post,wv) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens







import numpy as np

fname='abstract.csv'
df = pd.read_csv(fname,encoding='latin-1')
comtdata=df
test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r[0]), axis=1).values
X_comtdata_average1 = word_averaging_listn(model1,test_tokenized)
X_comtdata_average2 = word_averaging_listn(model2,test_tokenized)


X_comtdata_average1=np.concatenate((X_comtdata_average1, y.reshape(-1,1)), axis=1)
X_comtdata_average2=np.concatenate((X_comtdata_average2, y.reshape(-1,1)), axis=1)
fname='cbow.csv'
np.savetxt(fname,X_comtdata_average1, delimiter=',', fmt='%f')  
fname='skg.csv'
np.savetxt(fname,X_comtdata_average2, delimiter=',', fmt='%f')

