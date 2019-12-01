#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:56:39 2019

@author: jim
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
import re

import pdb


#%%
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import MaxPooling1D, Conv1D, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Input
from sklearn import preprocessing
from keras.utils import plot_model


#%%
'''
Just like Lab2 to use label encoding
'''
import keras
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('logs/training_log.csv')

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

def get_class_ratio(train_df):
    post_total = len(train_df)
    df1 = train_df.groupby(['emotion']).count()['text']
    df1 = df1.apply(lambda x: round(post_total/x,3))
    return df1

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%5 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("./logs/model_{}.hd5".format(epoch))


#%%
import joblib
import os

emo = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
# Read preprocess.csv
'''
Preprocessing here is just to extract and clean up the information from .json, such as 
tweet id and the text info.
Therefore I don't upload the code for preprocessing.
'''
train_df = pd.read_csv('./preprocess_train.csv', lineterminator='\n')
train_df['text'].fillna('OUTOFWORDS', inplace=True)
submit = pd.read_csv('./sampleSubmission.csv')
test_df = pd.read_csv('./preprocess_test.csv', lineterminator='\n')
test_df['text'].fillna('OUTOFWORDS', inplace=True)
# rearange the index to be the same as the submitted one
test_df = test_df.set_index('tweet_id')
test_df = test_df.reindex(index=submit['id'])
test_df = test_df.reset_index()
# Extract tf-idf
max_feat = 5000
min_df = 5
ngram_min = 1
ngram_max=3
vec_file = './vec_{}_{}_{}_{}.pkl'.format(max_feat, min_df, ngram_min, ngram_max)
# TFIDF
if not os.path.isfile(vec_file):
    print('vectorizer not exitst, so creating....')
    vectorizer = TfidfVectorizer(max_features=max_feat, min_df=min_df, ngram_range=(ngram_min, ngram_max), 
                                 tokenizer=None, stop_words=None)
    print('transforming..')
    train_tfidf = vectorizer.fit_transform(train_df['text'])
    test_tfidf = vectorizer.transform(test_df['text'])
    joblib.dump(vectorizer, './vec_{}_{}_{}_{}.pkl'.format(max_feat, min_df, ngram_min, ngram_max))
else:
    print('vectorizer exitst, so loading..')
    vectorizer = joblib.load(vec_file)
    print('transforming..')
    train_tfidf = vectorizer.transform(train_df['text'])
    test_tfidf = vectorizer.transform(test_df['text'])
    
LE = preprocessing.LabelEncoder()
LE.fit(train_df['emotion'].values)

y_train = label_encode(LE, train_df['emotion'].values)
# Build the model
tfidf_input = Input(shape=(max_feat, ), name='tfidf_input')

dense_tfidf_1 = Dense(256, name='tfidf_dense_1')(tfidf_input)
act_tfidf_1 = Activation('relu', name='tfidf_act_1')(dense_tfidf_1)
drop_tfidf_1 = Dropout(0.2, name='tfidf_drop_1')(act_tfidf_1)

dense_tfidf_2 = Dense(256, name='tfidf_dense_2')(act_tfidf_1)
act_tfidf_2 = Activation('relu', name='tfidf_act_2')(dense_tfidf_2)
drop_tfidf_2 = Dropout(0.2, name='tfidf_drop_2')(act_tfidf_2)

dense_tfidf_3 = Dense(128, name='tfidf_dense_3')(drop_tfidf_2)
act_tfidf_3 = Activation('relu', name='tfidf_act_3')(dense_tfidf_3)

dense_tfidf_4 = Dense(128, name='tfidf_dense_4')(act_tfidf_3)
act_tfidf_4 = Activation('relu', name='tfidf_act_4')(dense_tfidf_4)

dense_layer_out = Dense(len(LE.classes_), activation='sigmoid', name='output')(act_tfidf_4)
model = Model(inputs=tfidf_input, outputs=dense_layer_out)
# Save the model structure
plot_model(model, './kaggle.png')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


saver = CustomSaver()
history = model.fit(train_tfidf, y_train, batch_size=1024, epochs=30, verbose=1, callbacks=[saver])#, class_weight=dd)
prediction = model.predict(test_tfidf, batch_size=1024)
prediction = label_decode(LE, prediction)
# Fill in the predicted results to the submitted file
submit['emotion'] = prediction
submit.to_csv('./tfidf_dense.csv', index=None)