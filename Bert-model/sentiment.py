#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%
#%%
# load the data that has been cleaned
# 1. removed html tags, urls, punctuations
# 2. the sentences were first tokenized by TweetTokenizer from nltk
# 3. changed emoticons to 'happy' or 'sad'
# 4. lowered all the words and did the spell check
# 5. each tweet has been split into a bunch of words

train = pd.read_csv('cleaned_data/clean_train_v2.csv')
test = pd.read_csv('cleaned_data/clean_test_v2.csv')
target = train["target"]
train.drop(['id','keyword','location','target'], axis=1, inplace=True)
test.drop(['id','keyword','location'], axis=1, inplace=True)
ytest = np.array(pd.read_csv('nlp-getting-started/gt_test.csv')['target'])

#%%
nltk.download('vader_lexicon')

#%%
sid = SentimentIntensityAnalyzer()
for i, sentence in enumerate(train['text']):
    if i > 20:
        break
    print (sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print (f'{k}: {ss[k]}', end=' ')
    print ('\n')

#%%
def prepareSentimentData(df, sid):
    data = np.zeros((len(df), 4))
    for i, sentence in enumerate(df['text']):
        ss = sid.polarity_scores(sentence)
        ent = []
        for k in sorted(ss):
            ent.append(ss[k])
        data[i,:] = np.array(ent)
    return data

trainScores = prepareSentimentData(train,sid)
testScores = prepareSentimentData(test, sid)

#%%
