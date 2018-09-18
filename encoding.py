# coding: utf-8

import numpy as np

def encoding( data, encode_type, time_length, vocab_size ):
    if encode_type == '1hot':
        return onehot_encoding(data, time_length, vocab_size)
    elif encode_type == 'embedding':
        return data

def onehot_encoding( data, time_length, vocab_size):
    X = np.zeros((len(data), time_length, vocab_size), dtype=np.bool)
    for i, sent in enumerate(data):
        for j, k in enumerate(sent):
            X[i, j, k] = 1
    return X

def history_build(indata, pad_X):
    H = list()
    for i, pos in enumerate(indata.dataset['startid']):
        if i == pos:
            his = []
        else:
            his = list(pad_X[pos])
            for j in range(pos + 1, i):
                his += list(pad_X[j])
        
        H.append(list(his))
    return H

