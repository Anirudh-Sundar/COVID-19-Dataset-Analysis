# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np 


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

fname='abstract.csv'
df = pd.read_csv(fname,encoding='latin-1')
x = tfidf.fit_transform(df[0])
df_tfidf = pd.DataFrame(x.toarray())
x=np.array(df_tfidf)
y=df['label']
y=np.array(y)

f=tfidf.get_feature_names()
np.savez('./data/nfd1', f=f)


impw=np.zeros((np.shape(x)[1],1))
for i in range(0,np.shape(x)[1]):
    impw[i]=len(np.where(x[:,i]!=0)[0])
    
ino=np.where(impw>=10)
x=x[:,ino[0]]
x_sum=np.sum(x,axis=1)
ino=np.where(x_sum!=0)
x=x[ino[0],:]
y=y[ino[0]]
x=np.concatenate((x, y.reshape(-1,1)), axis=1)
fname='tfidfvalue.csv'
np.savetxt(fname,x, delimiter=',', fmt='%f')  






    

    
    