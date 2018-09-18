# coding: utf-8

import re
import numpy as np


class PredefinedEmbedding(object):
    """
     dictionary of embeddings
   """
    def __init__(self, embedding_file):
        self.embeddings = read_embeddings(embedding_file)
        self.embeddings['embeddings']['BOS'] = np.random.uniform(size = (self.embeddings['embedding_size'], 1))
        self.embeddings['embeddings']['EOS'] = np.random.uniform(size = (self.embeddings['embedding_size'], 1))
        
    def get_embedding_dim(self):
        return self.embeddings['embedding_size']
    
    def get_word_embedding(self, word):
#         if word == 'BOS':         #?? BOS? 
#             return self.embeddings['embeddings']["</s>"]     #?? </s> ??
        
#         elif word == 'EOS':       #?? EOS?
#             return self.embeddings['embeddings']["</s>"]
        
        if word in self.embeddings['embeddings']:
            return self.embeddings['embeddings'][word]
        
        else:
            return np.zeros((self.embeddings['embedding_size'], 1))             # for words like <pad>, <unk>

def read_embeddings(embedding_file):
    # read the word embeddings
    # each line has one word and a vector of embeddings listed as a sequence of real valued numbers
    word_embeddings = {}
    first = True
    p = re.compile(r'\s+')
    
    f = open(embedding_file)          # concepnet-numberbatch
    next(f)
    for line in f:
        d = p.split(line.strip())
        if first:
            first = False
            size = len(d) - 1
        else:
            if (size != len(d) -1):
                print(size, len(d) - 1)
                print("Problem with embedding file, not all vectors are the same length\n")
                return
        
        current_word = d[0]
        word_embeddings[current_word] = np.zeros((size,1))
        for i in range(1,len(d)):
            word_embeddings[current_word][i-1] = float(d[i])
    
    embeddings = {'embeddings':word_embeddings, 'embedding_size': size}
    
    return embeddings
