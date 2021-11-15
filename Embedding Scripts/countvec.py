# -*- coding: utf-8 -*-


import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

fname='abstract.csv'
df = pd.read_csv(fname,encoding='latin-1')
vectorizer = CountVectorizer(ngram_range=(1,2))
x = vectorizer.fit_transform(df[0])
x=x.toarray()

f=vectorizer.get_feature_names()
np.savez('./data/nfd', f=f)


impw=np.zeros((np.shape(x)[1],1))
for i in range(0,np.shape(x)[1]):
    impw[i]=len(np.where(x[:,i]!=0)[0])
    
ino=np.where(impw>=10)
x=x[:,ino[0]]
x_sum=np.sum(x,axis=1)
ino=np.where(x_sum!=0)
x=x[ino[0],:]



fname='coutv.csv'
np.savetxt(fname,x, delimiter=',', fmt='%f')  

