# coding: utf-8

import re

import numpy as np


# ### Convert pizza_dataset into this format
# 
# `words \t intent \t state \t tags \t start`

class dataSet(object):
    def __init__(self, dataFile, toggle, wordDictionary = None, tagDictionary = None, intent2idx = None, id2word = None, id2tag = None, intent_labels = []):
        if toggle == 'train':
            self.dataset = readData(dataFile, intent_labels)
        elif toggle == 'val' or toggle == 'test':
            if not intent_labels:
                raise Exception('Intent labels is not given')
            self.dataset = readTest(dataFile, wordDictionary, tagDictionary, intent2idx, id2word, id2tag, intent_labels)
        
        else:
            raise Exception('Toggle must be [train, val, test]')
    
    def getNum(numFile):
        return readNum(numFile)
    
    def getNoOfExamples(self):
        return self.dataset['uttCount']

    def getIntentLabels(self):
        return self.dataset['intent_labels']
    
    def getWordVocabSize(self):
        return self.dataset['wordVocabSize']

    def getIntentVocabSize(self):
        return self.dataset['intentVocabSize']
    
    def getTagVocabSize(self):
        return self.dataset['tagVocabSize']
    
    def getExampleUtterance(self, idx):
        self.dataset['utterances'][idx]
    
    def getExampleIntent(self, idx):
        self.dataset['intents'][idx]
    
    def getExampleTags(self, idx):
        self.dataset['tags'][idx]
        
    def getWordVocab(self):
        return self.dataset['word2id']

    def getIntentVocab(self):
        return self.dataset['intent2id']
    
    def getTagVocab(self):
         return self.dataset['tag2id']
    
    def getIndex2Word(self):
        return self.dataset['id2word']
    
    def getIndex2Tag(self):
        return self.dataset['id2tag']
    
    def getWordAtIndex(self, idx):
        return self.dataset['id2word'][idx]
    
    def getTagAtIndex(self, idx):
        return self.dataset['id2tag'][idx]
    
    def getSample(self, batch_size):
        inputs = {}
        targets = {}
        indices = np.random.randint(0, self.getNoExamples, size = batch_size)
        
        for i, ind in enumerate(indices):
            inputs[i] = self.getExampleUtterance(ind)
            targets[i] = [self.getExampleIntent(ind), self.getExampleTags(ind)]
        
        return inputs, targets


def readData(fileName, intent_labels):
    check = False
    if intent_labels:
        check = True
    else:
        intent_labels = []
        
    utterances = list()
    intents = list()
    tags = list()
    starts = list()
    startid = list()
    
    word_vocab_index = 1
    word2idx = {'<unk>' : 0}
    idx2word = ['<unk>']
    
    intent_vocab_index = 0
    intent2idx = {}

    tag_vocab_index = 1
    tag2idx = {'<unk>' : 0}
    idx2tag = ['<unk>']
    
    utt_count = 0
    tmp_startid = 0
    
    for line in open(fileName):
        d = line.split('\t')
        utt = d[0].strip()
        
        intent = d[1].strip()          # assuming intent classification is multi-class classification
        if intent not in intent_labels and check:
            print('Intent %s not found in intent_labels %s' % (intent, intent_labels))
            print('SKIPPING !!!!')
            continue

        elif intent not in intent_labels:
            intent_labels.append(intent) 
        
        if intent not in intent2idx:
            intent2idx[intent] = intent_vocab_index
            intent_vocab_index += 1

        intent = intent2idx[intent]
        intents.append(intent)

        tag = d[2].strip()
        
        if len(d) > 3:
            start = np.bool(int(d[3].strip()))
            starts.append(start)
            if start:
                tmp_startid = utt_count
                
            startid.append(tmp_startid)
            
        tmp_utt = list()
        tmp_tag = list()
        mywords = utt.split()
        mytags = tag.split()
        
        if len(mywords) != len(mytags):
            print(mywords)
            print(mytags)
            raise Exception('Length mismatch of sentence and token')
        
        for i in range(len(mywords)):
            if mywords[i] not in word2idx:
                word2idx[mywords[i]] = word_vocab_index
                idx2word.append(mywords[i])
                word_vocab_index += 1
            
            if mytags[i] not in tag2idx:
                tag2idx[mytags[i]] = tag_vocab_index
                idx2tag.append(mytags[i])
                tag_vocab_index += 1
        
            tmp_utt.append(word2idx[mywords[i]])
            tmp_tag.append(tag2idx[mytags[i]])
        
        
        utterances.append(tmp_utt)
        tags.append(tmp_tag)
        utt_count += 1
    
    data = {'start' : starts, 
            'startid' : startid, 
            'utterances' : utterances,
            'intents': intents,
            'tags' : tags,
            'uttCount' : utt_count,
            'id2word' : idx2word,
            'id2tag' : idx2tag,
            'wordVocabSize' : word_vocab_index,
            'tagVocabSize' : tag_vocab_index,
            'intentVocabSize' : intent_vocab_index,
            'word2id' : word2idx,
            'tag2id' : tag2idx,
            'intent2id' : intent2idx,
            'intent_labels' : intent_labels
           }
    
    return data


def readTest(fileName, word2idx, tag2idx, intent2idx, idx2word, idx2tag, intent_labels):
    utterances = list()
    intents = list()
    tags = list()
    starts = list()
    startid = list()
    
    utt_count = 0
    tmp_startid = 0
    
    for line in open(fileName):
        d = line.split('\t')
        utt = d[0].strip()
        intent = d[1].strip()
        tag = d[2].strip()
        
        if intent not in intent_labels:
            print('Intent %s not found in intent_labels %s' % (intent, intent_labels))
            print('SKIPPING !!!!')
            continue    
            
        intent = intent2idx[intent]
        intents.append(intent)
        
        if len(d) > 3:
            start = np.bool(int(d[3].strip()))
            starts.append(start)
            if start:
                tmp_startid = utt_count
            
            startid.append(tmp_startid)
        
        tmp_utt = list()
        tmp_tag = list()
        mywords = utt.split()
        mytags = tag.split()
        
        for i in range(len(mywords)):
            if mywords[i] not in word2idx:
                tmp_utt.append(word2idx['<unk>'])
            else:
                tmp_utt.append(word2idx[mywords[i]])
            
            if mytags[i] not in tag2idx:
                tmp_tag.append(tag2idx['<unk>'])
            else:
                tmp_tag.append(tag2idx[mytags[i]])
                
        utterances.append(tmp_utt)
        tags.append(tmp_tag)
        utt_count += 1
        
    wordVocabSize = len(word2idx)
    tagVocabSize = len(tag2idx)
        
    data = {'start' : starts, 
            'startid' : startid, 
            'utterances' : utterances,
            'intents': intents,
            'tags' : tags,
            'uttCount' : utt_count,
            'id2word' : idx2word,
            'id2tag' : idx2tag,
            'wordVocabSize' : len(word2idx),
            'tagVocabSize' : len(tag2idx),
            'intentVocabSize' : len(intent2idx),
            'word2id' : word2idx,
            'tag2id' : tag2idx,
            'intent2id' : intent2idx,
            'intent_labels' : intent_labels
           }
    
    return data


def readNum(numFile):
    numList = list(map(int, open(numFile).read().strip().split()))
    totalList = list()
    curr = 0
    
    for num in numList:
        curr += num + 1
        totalList.append(curr)
        
    return numList, totalList

