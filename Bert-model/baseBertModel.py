#%%
import numpy as np
import pandas as pd
import re
import nltk
import string
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split



#%%
# load the data that has been cleaned
# 1. removed html tags, urls, punctuations
# 2. the sentences were first tokenized by TweetTokenizer from nltk
# 3. changed emoticons to 'happy' or 'sad'
# 4. lowered all the words and did the spell check
# 5. each tweet has been split into a bunch of words

train = pd.read_csv('cleaned_data/clean_train.csv')
test = pd.read_csv('cleaned_data/clean_test.csv')
target = train["target"]
train.drop(['id','keyword','location','target'], axis=1, inplace=True)
test.drop(['id','keyword','location'], axis=1, inplace=True)
train['text'] = train['text'].apply(lambda tw: eval(tw))
test['text'] = test['text'].apply(lambda tw: eval(tw))

#%%
# load pretrained model from BERT
bertTknzr = BertTokenizer.from_pretrained('bert-base-uncased')
#%%
bertModel = BertModel.from_pretrained('bert-base-uncased')

#%%
# check the coverage ratio of bert vocabulary and our corpus
def checkCoverage(df, wordlist):
    wordset = []
    for sentence in df['text']:
        wordset.extend(sentence)
    wordset = set(wordset)
    wordlist = set(wordlist)
    overlap = wordlist.intersection(wordset)
    ratio = len(overlap) / len(wordset)
    left = len(wordset) - len(overlap)
    print (f"{ratio*100}% of words exist, the number of never seen is: {left}")
    return 

bert_wordlist = list(bertTknzr.vocab.keys())
print ('Training corpus:')
checkCoverage(train, bert_wordlist)
print ('Test corpus:')
checkCoverage(test, bert_wordlist)

# Training corpus:
# 60.12% of words exist, the number of never seen is: 5641
# Test corpus:
# 67.21% of words exist, the number of never seen is: 2891

#%%
def bertEncoder(dftrain, dftest):
    tweets = dftrain['text'].apply(lambda tw: " ".join(tw))
    tokenized = tweets.apply(lambda tw: bertTknzr.encode(tw, add_special_tokens=True))
    max_len = 0
    for i in tokenized.values:
        max_len = max(max_len, len(i))
    padded_train = np.array([tokens + [0] * (max_len - len(tokens)) for tokens in tokenized.values])
    
    tweets = dftest['text'].apply(lambda tw: " ".join(tw))
    tokenized = tweets.apply(lambda tw: bertTknzr.encode(tw, add_special_tokens=True))
    padded_test = np.array([tokens + [0] * (max_len - len(tokens)) for tokens in tokenized.values])

    return padded_train, padded_test

trainEncoded, testEncoded = bertEncoder(train, test)

#%%
trainMask = np.where(trainEncoded != 0, 1, 0)
testMask = np.where(testEncoded != 0, 1, 0)

#%%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenIdx, masks, targets):
        super(Dataset, self).__init__()
        self.tokenIdx = tokenIdx
        self.masks = masks
        self.targets = targets

    def __getitem__(self, i):
        tokens = self.tokenIdx[i]
        mask = self.masks[i]
        target = self.targets[i]
        return tokens, mask, target
    
    def __len__(self):
        return len(self.tokenIdx)

permu = np.random.permutation(len(trainEncoded))
Xtrain = (trainEncoded[permu[:6400], :], trainMask[permu[:6400], :])
ytrain = np.array(target[permu[:6400]])
Xval = (trainEncoded[permu[6400:],:], trainMask[permu[6400:], :])
yval = np.array(target[permu[6400:]])
Xtest = (testEncoded, testMask)

training_dataset = Dataset(Xtrain[0], Xtrain[1], ytrain)
val_dataset = Dataset(Xval[0], Xval[1], yval)

#%%
class model_1(torch.nn.Module):
    def __init__(self, bert):
        super(model_1, self).__init__()
        self.bertBase = bert
        self.fc = torch.nn.Linear(768, 2)
        self.logSoftmax = torch.nn.LogSoftmax()
    
    def forward(self, idx, mask):
        x = self.bertBase(idx, attention_mask=mask)[0]
        x = self.fc(x[:,0,:])
        x = self.logSoftmax(x)
        return x

model = model_1(bertModel)

#%%
trainLoader = DataLoader(dataset=training_dataset, batch_size=1, shuffle=True)
valLoader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)
criterion = torch.nn.NLLLoss()
learning_rate = 0.001
max_iter = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lossHis = []
valHis = []

#%%
def checkLossVal(model):
    lossSum = 0.0
    cnt = 0
    for tokens, masks, target in valLoader:
        y_pre = model(tokens.long(), masks.long())
        loss = criterion(y_pre, torch.LongTensor(target))
        lossSum += loss.item()
        cnt += 1
        break
    return lossSum/cnt

#%%
for t in tqdm(range(max_iter)):
    lossSum = 0.0
    i = 0
    for tokens, masks, y in trainLoader:
        y_pre = model(tokens.long(), masks.long())
        loss = criterion(y_pre, torch.LongTensor(y))
        lossSum += loss.item()
        i += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    lossSum /= i
    lossHis.append(lossSum)
    print("epoc: "+str(t)+" MSE: "+str(lossSum))

    if t%30 == 0:
        try:
            torch.save(model, 'train-log/epoch'+str(t)+'.pkl')
        except FileNotFoundError:
            print ("Can't find the file!")
        valHis.append(checkLossVal(model))
        print('Validation Loss: ' + str(valHis[-1]))



#%%

#%%
