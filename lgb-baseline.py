#%%
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import nltk
import html
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import Counter

#%% 
# load stop words from nltk
stop_words = stopwords.words('english')


#%%
# read training and test data
train = pd.read_csv('nlp-getting-started/train.csv')
test = pd.read_csv('nlp-getting-started/test.csv')
target = train["target"]
train.drop(['id','keyword','location','target'], axis=1, inplace=True)
test.drop(['id','keyword','location'], axis=1, inplace=True)

#%% 
# extract 7 handcraft features before data cleaning procedure
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
# data cleaning
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_punc(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def token(text):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    words = [word for word in tknzr.tokenize(text) \
            if ((word not in stop_words) and \
                (word.isalpha() is True))]
    return ' '.join(words)
    

def textClean(df):
    df['text'] = df['text'].apply(lambda text: html.unescape(text))
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda text: remove_URL(text))
    df['text'] = df['text'].apply(lambda text: remove_html(text))
    df['text'] = df['text'].apply(lambda text: token(text))
    df['text'] = df['text'].apply(lambda text: remove_punc(text))
    return df

train = textClean(train)
test = textClean(test)

#%%
# build corpus (list of words of each tweet) from cleaned data 
def buildCorpus(df):
    corpus = []
    for tweet in df['text']:
        words = [word for word in tweet.split()]
        corpus.append(words)
    return corpus

train_corpus = buildCorpus(train)
test_corpus = buildCorpus(test)


#%% 
# embedding words using pretrained 100d-GloVec
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
# Try flat feature/ feature prepare/ train validation split
YTrainVal = np.array(pd.read_csv('nlp-getting-started/train.csv')['target'])
XTrainVal = np.hstack((trainValWordFeat.reshape(train_handcraft.shape[0],-1), np.array(train_handcraft)))
Xtest = np.hstack((testWordFeat.reshape(test_handcraft.shape[0],-1), np.array(test_handcraft)))
Xtrain, Xval, ytrain, yval = train_test_split(XTrainVal, YTrainVal, test_size=0.15)


#%%
# build gradientBoosting model
params = {"learning_rate": 0.05,
          "max_depth": 20,
         "boosting": "gbdt",
         "bagging_freq": 1,
         "bagging_fraction": 0.95,
          "colsample_bytree": 0.5,
         "min_data_in_leaf": 50,
         "bagging_seed": 42,
          "lambda_l2": 0.0001,
         "metric": "binary_logloss",
         "random_state": 42}

def evalAcc(pred, dataset):
    gt = dataset.label
    pred = (pred > 0.5).astype(int)
    acc = np.sum(gt == pred)/len(pred)
    return "Accuracy",acc,True

train_data = lgb.Dataset(Xtrain, label=ytrain)
val_data = lgb.Dataset(Xval, label=yval)
clf = lgb.train(params, train_data, num_boost_round=10000,\
     valid_sets = [train_data, val_data], verbose_eval=100, early_stopping_rounds=100, feval=evalAcc)


#%%
# predict on test set and write predictions into file

def writePredictions(y, filename="submission.csv"):
    outpred = pd.read_csv("nlp-getting-started/sample_submission.csv")
    outpred["target"] = y
    outpred.to_csv(filename,index=False)
    return 

ypred = (clf.predict(Xtest) > 0.5).astype(int)
writePredictions(ypred)


#%%
ytest = np.array(pd.read_csv("nlp-getting-started/gt_test.csv")['target'])

#%%
np.sum(ypred == ytest)/len(ytest)