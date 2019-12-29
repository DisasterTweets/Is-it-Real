#%%
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import lightgbm as lgb
import nltk
import html
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter

#%% 
stop_words = stopwords.words('english')


#%%
train = pd.read_csv('nlp-getting-started/train.csv')
test = pd.read_csv('nlp-getting-started/test.csv')
target = train["target"]
train.drop(['id','keyword','location','target'], axis=1, inplace=True)
test.drop(['id','keyword','location'], axis=1, inplace=True)

#%% 
def handcraftFeat(df):
    basicFeat = pd.DataFrame()
    basicFeat['num_char'] = df['text'].apply(len)
    basicFeat['num_words'] = df['text'].apply(lambda text: len(text.split()))

    counts = df['text'].apply(lambda text: Counter(text))
    basicFeat['num_hashtag'] = counts.apply(lambda c: 0 if '#' not in c else c['#'])
    basicFeat['num_mention'] = counts.apply(lambda c: 0 if '@' not in c else c['@'])
    basicFeat['num_exclamation'] = counts.apply(lambda c: 0 if '!' not in c else c['!'])
    basicFeat['num_question'] = counts.apply(lambda c: 0 if '?' not in c else c['?'])
    basicFeat['has_url'] = df['text'].apply(lambda text: 1 if 'http' in text else 0)

    return basicFeat

train_handcraft = handcraftFeat(train)
test_handcraft = handcraftFeat(test)

#%% 

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_punc(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def textClean(df):
    df['text'] = df['text'].apply(lambda text: html.unescape(text))
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda text: remove_URL(text))
    df['text'] = df['text'].apply(lambda text: remove_html(text))
    df['text'] = df['text'].apply(lambda text: remove_punc(text))
    return df

train = textClean(train)
test = textClean(test)

#%%
def buildCorpus(df):
    tknzr = TweetTokenizer()
    corpus = []
    for tweet in df['text']:
        words = [word for word in tknzr.tokenize(tweet) \
                if ((word not in stop_words) and \
                    (word.isalpha() is True))]
        corpus.append(words)
    return corpus

train_corpus = buildCorpus(train)
test_corpus = buildCorpus(test)


#%% 
embedding_dict = {}
with open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding='utf8') as f:
    for line in f:
        #line = line.encode('utf-8')
        values = line.split()
        word = values[0].replace('<','').replace('>','')
        if word.isalpha() is False:
            continue
        vec = np.array(values[1:], dtype='float32')
        embedding_dict[word] = vec

#%%
punctuation_dict = {}
with open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0].replace('<','').replace('>','')
        if word not in string.punctuation:
            continue
        vec = np.array(values[1:], dtype='float32')
        punctuation_dict[word] = vec

#%%
def buildDataset(corpus, dic=embedding_dict, vec_length=100, padding_length=25):
    num_tweets = len(corpus)
    dataset = np.zeros((num_tweets, padding_length, vec_length))
    for tid, tweet in enumerate(corpus):
        idx = 0
        for word in tweet:
            vec = embedding_dict.get(word)
            if vec is not None:
                dataset[tid, idx, :] = vec.reshape((1, -1))
                idx += 1
                if idx >= padding_length:
                    break
    return dataset

trainValWordFeat = buildDataset(train_corpus, embedding_dict)
testWordFeat = buildDataset(test_corpus, embedding_dict)
            
#%% 
trainValY = np.array(pd.read_csv('nlp-getting-started/train.csv')['target'])

#%%
# Try flat feature
XtrainVal = 

#%%
np.array(train_handcraft)