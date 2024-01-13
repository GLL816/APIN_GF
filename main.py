import numpy as np
import os
import sys
import wave
import copy
import math

#from __future__ import print_function
from keras.models import *
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward

import scipy.io as sio
from keras.models import Sequential, Model,load_model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, GRU, Input, Flatten, Add, Embedding, Convolution1D, MaxPooling1D, Dropout, Conv1D, concatenate, Conv2D, MaxPool2D, Reshape, Bidirectional, TimeDistributed
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
#import pandas as pd
from numpy import genfromtxt
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, Concatenate,Permute, Multiply
from keras.backend.tensorflow_backend import set_session
import keras.backend as bk
from keras.backend.tensorflow_backend import set_session
import keras.backend.tensorflow_backend as KTF
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import backend as K
#import tensorflow_hub as hub
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
#import seaborn as sns
import keras.layers as layers
#from urllib.request import urlopen
import codecs
from keras.callbacks import EarlyStopping, TensorBoard
import csv
from math import*
from decimal import Decimal
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
import random
#from scipy.special import softmax
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Permute
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras.layers import Multiply
import h5py
from keras.utils import np_utils
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
KTF.set_session(session)

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

LR = 0.001
batch_size = 40
num_classes = 4
epochs =30
batch_index=0
CELL_SIZE =128
TIME_STEPS = 400    
INPUT_SIZE = 1024
head=32

class APIN(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(APIN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(APIN, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
       
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class GFusion(Layer):
    def __init__(self, **kwargs):
        super(GFusion, self).__init__(** kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W1 = self.add_weight(name='att_weight11', 
                                 shape=(input_shape[0][1], input_shape[0][1]),
                                 initializer='uniform',
                                 trainable=True)
        self.W2 = self.add_weight(name='att_weight21', 
                                 shape=(input_shape[1][1], input_shape[1][1]),
                                 initializer='uniform',
                                 trainable=True)
        
        self.b1 = self.add_weight(name='att_bias1', 
                                 shape=(input_shape[0][1],),
                                 initializer='uniform',
                                 trainable=True)
        
        self.b2 = self.add_weight(name='att_bias2',
                                 shape=(input_shape[1][1],),
                                 initializer='uniform',
                                 trainable=True)
        
        super(GFusion, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        x1 = K.permute_dimensions(inputs[0], (0, 2, 1)) #speech
        x2 = K.permute_dimensions(inputs[1], (0, 2, 1))
        g1 = K.tanh(K.dot(x1, self.W1) + self.b1)  #tanh
        g2 = K.tanh(K.dot(x2, self.W2) + self.b2)
        #g = K.sigmoid(K.dot(x1, a1)+K.dot(x2, a2))

        #g = K.dot(x1, g1)+K.dot(x2, g2)
        g = g1*x1+g2*x2
        #g = K.sigmoid(K.dot(x1, self.W1)+K.dot(x2, self.W2))
        gF = K.permute_dimensions(g, (0, 2, 1))
        return gF  #gated feature

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2]


f1=h5py.File('/Work21/2016/guolili/CFP2023/trainRNN/A5.mat')
x1_train=np.transpose(f1['train_x'])
x1_test=np.transpose(f1['test_x'])
y_train=np.transpose(f1['train_y'])
y_test=np.transpose(f1['test_y'])

f2=h5py.File('/Work21/2016/guolili/CFP2023/trainRNN/DRP5.mat')
x2_train=np.transpose(f2['train_DRP'])
x2_test=np.transpose(f2['test_DRP'])

print('x_train shape:', x1_train.shape)
print(x1_train.shape[0], 'train samples')
print(x1_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, num_classes=4)
y_test = np_utils.to_categorical(y_test, num_classes=4)     

visible1 = Input(shape=(TIME_STEPS, INPUT_SIZE), name='input1') 
h1 = Bidirectional(LSTM(CELL_SIZE, dropout=0.2, return_sequences=True))(visible1)
h1 = Bidirectional(LSTM(CELL_SIZE, dropout=0.2, return_sequences=True))(h1)

visible2 = Input(shape=(TIME_STEPS, INPUT_SIZE), name='input2') 
h2 = Bidirectional(LSTM(CELL_SIZE, dropout=0.2, return_sequences=True))(visible2)
h2 = Bidirectional(LSTM(CELL_SIZE, dropout=0.2, return_sequences=True))(h2)

# A-P interaction
ad=APIN(head,2*CELL_SIZE//head)([h1,h2,h2])
ad = Add()([ad, h1])
ad = LayerNormalization()(ad)

da=APIN(head,2*CELL_SIZE//head)([h2,h1,h1])
da = Add()([da, h2])
da = LayerNormalization()(da)
# Gated fusion
merge = GFusion()([ad, da])
merge = AttentionLayer()(merge)
merge = Dense(256,activation='relu',name='getlayer')(merge)
merge = Dropout(0.5)(merge)
output = Dense(4, activation='softmax', name='main_output')(merge)
model = Model(inputs=[visible1, visible2], outputs=output)
model.summary()

# save model
filepath = "/Work21/2016/guolili/CFP2023/model/APT2_AD2-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss="categorical_crossentropy",
				optimizer="Adadelta",
				metrics=['accuracy'])

model.fit(x={'input1': x1_train, 'input2': x2_train}, y={'main_output': y_train},
		batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
		verbose=2,
		validation_data=([x1_test, x2_test], y_test))

score = model.evaluate(x={'input1': x1_test, 'input2': x2_test}, y={'main_output': y_test}, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
   
# load model
'''
model.load_weights('/Work21/2016/guolili/CFP2023/model/APT_AM-03-0.47.hdf5')
model.compile(loss="categorical_crossentropy",
				optimizer="Adadelta",
				metrics=['accuracy'])

score = model.evaluate(x={'input1': x1_test, 'input2': x2_test, 'input3': x3_test}, y={'main_output': y_test}, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#predict_labels = model.predict(x={'input1': x1_test, 'input2': x2_test, 'input3': x3_test})
#str2="sio.savemat('/Work21/2016/guolili/CFP2023/result/ALL.mat',{'predict_labels':predict_labels})"
#eval(str2)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(name='getlayer').output)
train_output = intermediate_layer_model.predict(x={'input1': x1_train, 'input2': x2_train, 'input3': x3_train})
test_output = intermediate_layer_model.predict(x={'input1': x1_test, 'input2': x2_test, 'input3': x3_test})
print('output shape:',test_output.shape)
sio.savemat('/Work21/2016/guolili/CFP2023/Toutput/AM.mat',{'train_data':train_output,'test_data':test_output})
'''
