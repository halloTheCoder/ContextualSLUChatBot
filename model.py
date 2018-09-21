
# coding: utf-8

# In[4]:


import os
import sys
import json
import argparse

from scipy import io
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, RepeatVector, Permute, Reshape, Activation, TimeDistributed, merge
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras import optimizers
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.constraints import nonneg, maxnorm

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


# In[3]:


from encoding import history_build, encoding
from word_slot_dataset import dataSet, readNum
from pre_defined_embedding import PredefinedEmbedding
from History import LossHistory


# In[5]:


class KerasModel(object):
    def __init__(self, argparams):
        # PARAMETERS
        self.hidden_size = argparams['hidden_size'] # size of hidden layer of neurons 
        self.learning_rate = argparams['learning_rate']
        self.training_file = argparams['train_data_path']
        self.validation_file = argparams['dev_data_path']
        self.test_file = argparams['test_data_path']
        self.result_path = argparams['result_path']
        self.update_f = argparams['sgdtype'] # options: adagrad, rmsprop, vanilla. default: vanilla      # optimizer
        self.decay_rate = argparams['decay_rate'] # for rmsprop
        self.default = argparams['default_flag'] # True: use defult values for optimizer
        self.momentum = argparams['momentum'] # for vanilla update
        self.max_epochs = argparams['max_epochs']
        self.activation = argparams['activation_func'] # options: tanh, sigmoid, relu. default: relu
        self.smooth_eps = argparams['smooth_eps'] # epsilon smoothing for rmsprop/adagrad/adadelta/adam/adamax 
        self.batch_size = argparams['batch_size']
#         self.input_type = argparams['input_type'] # options: 1hot, embedding, predefined
        
        self.emb_dict = argparams['embedding_file']
        self.embedding_size = argparams['embedding_size']
        self.dropout = argparams['dropout']
        self.dropout_ratio = argparams['dropout_ratio']
        self.iter_per_epoch = argparams['iter_per_epoch']
#         self.arch = argparams['arch']      # architecture to use, default LSTM
        self.init_type = argparams['init_type']
        self.fancy_forget_bias_init = argparams['forget_bias']
        self.time_length = argparams['time_length']
        self.his_length = argparams['his_length']
        self.mdl_path = argparams['mdl_path']
        self.log = argparams['log']
        self.record_epoch = argparams['record_epoch'] 
        self.load_weight = argparams['load_weight']
        self.shuffle = argparams['shuffle']
#         self.set_batch = argparams['set_batch']
        self.tag_format = argparams['tag_format']
        self.output_att = argparams['output_att']
        
        self.input_type = 'embedding' 
        self.arch = 'mem2n-r-blstm'
        
        self.model_arch = self.arch
        
        if self.validation_file is None:
            self.nodev = True
        else:
            self.nodev = False
            
        if self.input_type == 'embedding':
            self.model_arch = self.model_arch + '+emb'
        
        
    def test(self, H, X, data_type, tagDict, pad_data):
        # open a dir to store results
        if self.default:
            target_file = self.result_path + '/' + self.model_arch + '_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type
        else:
            target_file = self.result_path + '/' + self.model_arch +'-LR-'+str(self.learning_rate)+'_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type

        if 'memn2n' in self.arch or self.arch[0] == 'h':
            batch_data = [H, X]
        else:
            batch_data = X

        # output attention
        if self.output_att is not None:
            x1 = self.model.inputs[0]
            x2 = self.model.inputs[1]
            #x = self.model.layers[1].input
            y = self.model.get_layer(name='match').output
            #y = self.model.layers[9].output
            f = K.function([x1, x2, K.learning_phase()], y)
            att_mtx = f([batch_data[0], batch_data[1], 0])
            row, col = np.shape(att_mtx)
            fo = open(self.output_att, 'wb')
            for i in range(0, row):
                for j in range(0, col):
                    fo.write("%e " %att_mtx[i][j])
                fo.write('\n')
            fo.close()
            sys.stderr.write("Output the attention weights in the file %s.\n" %self.output_att)
            exit()
        if "predict_classes" in dir(self.model):
            prediction = self.model.predict_classes(batch_data)
            probability = self.model.predict_proba(batch_data)
        else:
            probability = self.model.predict(batch_data)
            prediction = np.argmax(probability, axis=2)

        # output prediction and probability results
        fo = open(target_file+"."+ data_type, "wb")
        for i, sent in enumerate(prediction):
            for j, tid in enumerate(sent):
                if pad_data[i][j] != 0:
                    if self.tag_format == 'normal':
                        fo.write(tagDict[tid] + ' ')
                    elif self.tag_format == 'conlleval':
                        fo.write(tagDict[tid] + '\n')
            fo.write('\n')
        fo.close()
        fo = open(target_file+"."+ data_type+'.prob', "wb")
        for i, sent in enumerate(probability):
            for j, prob in enumerate(sent):
                if pad_data[i][j] != 0:
                    for k, val in enumerate(prob):
                        fo.write("%e " %val)
                    fo.write("\n")
        fo.close()
        
        
    def build(self):
        # define optimizer function
        opt_func = self.update_f

        if not self.default:
            if self.update_f == 'sgd':
                opt_func = optimizers.SGD(lr = self.learning_rate, momentum = self.momentum, decay = self.decay_rate)
            elif self.update_f == 'rmsprop':
                opt_func = optimizers.RMSprop(lr = self.learning_rate, rho = self.rho, epsilon = self.smooth_eps)
            elif self.update_f == 'adagrad':
                opt_func = optimizers.Adagrad(lr = self.learning_rate, epsilon = self.smooth_eps)
            elif self.update_f == 'adadelta':
                opt_func = optimizers.Adadelta(lr=self.learning_rate, rho=self.rho, epsilon=self.smooth_eps)
            elif self.update_f == 'adam':
                opt_func = optimizers.Adam(lr = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2, epsilon = self.smooth_eps)
            elif self.update_f == 'adamax':
                opt_func = optimizers.Adamax(lr = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2, epsilon = self.smooth_eps)
            else:
                sys.stderr.write("Invalid optimizer.\n")
                exit()
        
        # memn2n-r-blstm
        raw_current = Input(shape = (self.time_length, ), dtype = 'int32', name = 'raw_current')
        current = Embedding(input_dim = self.input_vocab_size, output_dim = self.output_vocab_size, input_length = self.time_length, mask_zero = True)(raw_current)
        # current: (None, time_length, embedding_size)
        
        fcur_vec = LSTM(self.embedding_size, activation = self.activation, kernel_initializer = self.init_type, return_sequences = False)(current)
        bcur_vec = LSTM(self.embedding_size, activation = self.activation, kernel_initializer = self.init_type, return_sequences = False, go_backwards = True)(current)
        cur_vec = merge([fcur_vec, bcur_vec], mode = 'concat', concat_axis = -1)     # (None, 2 * embedding_size)
        
        sent_model = Model(inputs = raw_current, outputs = cur_vec)
        
        # apply the same function for mapping word sequences into sentence vecs
        # input_memory: (None, his_length, time_length)
        raw_input_memory = Input(shape = (self.his_length * self.time_length, ), dtype = 'int32', name = 'input_memory')
        input_memory = Reshape(target_shape = (self.his_length, self.time_length))(raw_input_memory)
        
        mem_vec = TimeDistributed(sent_model)(input_memory)       # (None, his_length, 2 * embedding_size)
        
        # compute the similarity between sentence embeddings for attention
        # cur_vec_extend = RepeatVector(self.his_length)(cur_vec)
        # nn.Linear(mem_vec + cur_vec)               # (None, his_length, )
        match = merge([mem_vec, cur_vec], mode = 'dot', dot_axes = [2, 1])
        match = Activation(activation = 'softmax', name = 'match')(match)       # (None, his_length)
        
        # encode the history with the current utterance and then feed into each timestep for tagging
        his_vec = merge([mem_vec, match], mode = 'dot', dot_axes = [1, 1])      # (None, 2 * embedding_size)
        o_vec = merge([his_vec, cur_vec], mode = 'sum')                         # (None, 2 * embedding_size)
        o_vec = Dense(self.embedding_size)(o_vec)
        o_vec = RepeatVector(self.time_length)(o_vec)                           # (None, time_len, embedding_size) this is the 'o' vector mentioned in the paper
        
        current_o = merge([current, o_vec], mode = 'concat', concat_axis = -1)
        print(current_o.shape)
        # fencoder = LSTM(self.hidden_size, return_sequences=False, kernel_initializer = self.init_type, activation=self.activation)(current_o)
        # bencoder = LSTM(self.hidden_size, return_sequences=False, kernel_initializer = self.init_type, activation=self.activation, go_backwards=True)(current_o)
        flabeling = LSTM(self.hidden_size, return_sequences=True, kernel_initializer = self.init_type, activation=self.activation)(current_o)
        blabeling = LSTM(self.hidden_size, return_sequences=True, kernel_initializer = self.init_type, activation=self.activation, go_backwards=True)(current_o)
        
        # encoder = merge([fencoder, bencoder], mode = 'concat', concat_axis = -1)
        encoder = merge([flabeling[:, -1, :], blabeling[:, -1, :]], mode = 'concat', concat_axis = -1)
        print(encoder.shape)
        tagger = merge([flabeling, blabeling], mode = 'concat', concat_axis = -1)
        print(tagger.shape)

        intent_pred = Dense(self.output_intent_size, activation = 'softmax', name = 'intent')(encoder)
        # encoder = RepeatVector(self.time_length)(encoder)
        # tagger = merge([encoder, labeling], mode = 'concat', concat_axis = -1)
        if self.dropout:
            tagger = Dropout(self.dropout_ratio)(tagger)
        
        prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'), name = 'slots')(tagger)

        self.model = Model(inputs = [raw_input_memory, raw_current], outputs = [intent_pred, prediction])
        self.model.compile(loss = 'categorical_crossentropy', optimizer = opt_func)
        
        
    def train(self, H_train, X_train, y_train, H_dev, X_dev, y_dev, val_ratio=0.0):
        # load saved model weights
        if self.load_weight is not None:
            sys.stderr.write("Loading the pretrained weights for the model.\n")
            self.model.load_weights(self.load_weight)
        else:
            # training batch preparation
            batch_train = [H_train, X_train]
            batch_dev = [H_dev, X_dev]
                
            # model training
            if not self.nodev:
                early_stop = EarlyStopping(monitor='val_loss', patience=10)
                train_log = LossHistory()
                self.model.fit(batch_train, y_train, batch_size = self.batch_size, epochs = self.max_epochs, verbose = 1, validation_data = (batch_dev, y_dev), callbacks = [early_stop, train_log], shuffle = self.shuffle)
                if self.log is not None:
                    fo = open(self.log, "wb")
                    for loss in train_log.losses:
                        fo.write("%lf\n" %loss)
                    fo.close()
            else:
                self.model.fit(batch_train, y_train, batch_size = self.batch_size, epochs = self.max_epochs, verbose = 1, shuffle = self.shuffle)
                    
    def run(self):
        # initializing the vocabularies
        trainData = dataSet(self.training_file, 'train')
        # print(trainData.getIntentLabels())
        testData = dataSet(self.test_file, 'test', trainData.getWordVocab(), trainData.getTagVocab(), trainData.getIntentVocab(), trainData.getIndex2Word(), trainData.getIndex2Tag(), trainData.getIntentLabels())
        
        intent_target_file = self.result_path + '/' + 'intent.list'
        with open(intent_target_file, 'w') as f:
            for intent in trainData.getIntentLabels():
                f.write(f"{intent}\n")
        
        tag_target_file = self.result_path + '/' + 'tag.list'
        with open(tag_target_file, 'w') as f:
            for tag in trainData.getIndex2Tag():
                f.write(f"{tag}\n")
        
        # preprocessing by padding 0 until maxlen
        X_train = sequence.pad_sequences(trainData.dataset['utterances'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
        X_test = sequence.pad_sequences(testData.dataset['utterances'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
        
        y_intent_train = trainData.dataset['intents']
        pad_y_tags_train = sequence.pad_sequences(trainData.dataset['tags'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
        
        y_intent_test = testData.dataset['intents']
        pad_y_tags_test = sequence.pad_sequences(testData.dataset['tags'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
        
        num_sample_train, max_len = np.shape(X_train)
        num_sample_test, _ = np.shape(X_test)
        
        if not self.nodev:
            validData = dataSet(self.validation_file, 'val', trainData.getWordVocab(), trainData.getTagVocab(), trainData.getIntentVocab(), trainData.getIndex2Word(), trainData.getIndex2Tag(), trainData.getIntentLabels())
            X_dev = sequence.pad_sequences(validData.dataset['utterances'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
            y_intent_dev = validData.dataset['intents']
            pad_y_tag_dev = sequence.pad_sequences(validData.dataset['tags'], maxlen = self.time_length, dtype = 'int32', padding = 'pre')
            num_sample_dev, _ = np.shape(X_dev)
        
        # encoding input vectors
        self.input_vocab_size = trainData.getWordVocabSize()
        self.output_intent_size = trainData.getIntentVocabSize()
        self.output_vocab_size = trainData.getTagVocabSize()
        
        print('Building model architecture!!!!')
        self.build()
        print(self.model.summary())
        
        # data generation
        sys.stderr.write("Vectorizing the input.\n")
        y_intent_train = to_categorical(y_intent_train, num_classes = self.output_intent_size)
        y_tags_train = encoding(pad_y_tags_train, '1hot', self.time_length, self.output_vocab_size)
        
        if not self.nodev:
            y_intent_dev = to_categorical(y_intent_dev, num_classes = self.output_intent_size)
            y_tags_dev = encoding(pad_y_tag_dev, '1hot', self.time_length, self.output_vocab_size)
        
        # encode history for memory network
        H_train = sequence.pad_sequences(history_build(trainData, X_train), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
        H_test = sequence.pad_sequences(history_build(testData, X_test), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
        if not self.nodev:
            H_dev = sequence.pad_sequences(history_build(validData, X_dev), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
        
        if self.record_epoch != -1 and self.load_weight is None:
            total_epochs = self.max_epochs
            self.max_epochs = self.record_epoch
            for i in range(1, total_epochs / self.record_epoch + 1):
                num_iter = i * self.record_epoch
                self.train(H_train=H_train, X_train=X_train, y_train=[y_intent_train, y_tags_train], H_dev=H_dev, X_dev=X_dev, y_dev=[y_intent_dev, y_tags_dev])
                if not self.nodev:
                    self.test(H=H_dev, X=X_dev, data_type='dev.'+str(num_iter),tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_dev)
                self.test(H=H_test, X=X_test, data_type='test.'+str(num_iter), tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_test)
                # save weights for the current model
                whole_path = self.mdl_path + '/' + self.model_arch + '.' + str(num_iter) + '.h5'
                sys.stderr.write("Writing model weight to %s...\n" %whole_path)
                self.model.save_weights(whole_path, overwrite=True)
        else:
            self.train(H_train=H_train, X_train=X_train, y_train=[y_intent_train, y_tags_train], H_dev=H_dev, X_dev=X_dev, y_dev=[y_intent_dev, y_tags_dev])
            
            # if not self.nodev:
            #     self.test(H=H_dev, X=X_dev, data_type='dev', tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_dev)
            
            # self.test(H=H_test, X=X_test, data_type='test', tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_test)
            
            with open('model.json') as f:
                json.dump(f, self.model.to_json())
            
            if self.load_weight is None:
                whole_path = self.mdl_path + '/' + self.model_arch + '.final-' + str(self.max_epochs) + '.h5'
                sys.stderr.write("Writing model weight to %s...\n" %whole_path)
                self.model.save_weights(whole_path, overwrite=True)

