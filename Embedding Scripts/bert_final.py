
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import torch
import transformers as ppb 


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text







model_class, tokenizer_class, pretrained_weights = (ppb.BertModel,ppb.BertTokenizer, 'bert-base-cased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

df = pd.read_csv('abstract.csv',encoding='latin-1')
df[0] = df[0].apply(clean_text)
print(df.head(10))


train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t), df['0']))

train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids,train_tokens))

b=np.zeros((np.shape(df)[0]))
for i in range(0,np.shape(df)[0]):
    b[i]=np.shape(train_tokens_ids[i])[0]
    
    
pad=np.zeros((np.shape(df)[0],int(np.max(b))))
for i in range(0,np.shape(df)[0]):
    a=train_tokens_ids[i]
    for j in range(0,np.shape(a)[0]):
        pad[i,j]=a[j]
        

for i in range(0,19):
    pad1=pad[24772:24774,:]       
    input_ids = torch.tensor(np.array(pad1))        
    with torch.no_grad():
        last_hidden_states = model(input_ids.long())    
        features = last_hidden_states[0][:,0,:].numpy()
        fname='D:/data/msr2013-bug_dataset-master/bertThunderbird'+str(19)+'.csv'
        np.savetxt(fname,features, delimiter=',', fmt='%f') 
