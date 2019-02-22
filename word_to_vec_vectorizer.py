# -*- coding: utf-8 -*-
#
# author: amanul haque
#

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.sklearn_api import W2VTransformer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import confusion_matrix, classification_report
import sys

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
            

class vectorization:

    def __init__(self):
          self.x = 0
          
    def get_test_train_split(self, X, y):
        labelled_set = np.array(range(4))
        unlabelled_set = np.array(range(4,len(y)))
        y[unlabelled_set] = -1
        
        return X, y, labelled_set, unlabelled_set

    def get_data(self, file_path):
                  
        data = pd.read_excel(file_path, sep=',')
        #print(data['text'])
        X = np.array(data['text'])
        y = np.array(data['label'])
        
        return X, y
    
    def vectorize_text(self, X, y, labelled_set, unlabelled_set):
        
        #X, y, labelled_set, unlabelled_set = self.get_test_train_split(X, y) 
        #print(labelled_set, X[labelled_set], y[labelled_set])
        #print(unlabelled_set, X[unlabelled_set], y[unlabelled_set])  
        
        filename = 'GoogleNews-vectors-negative300.bin'
        #model = KeyedVectors.load_word2vec_format(filename, binary=True)
        model = Word2Vec.load("w2v_model")
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        
        X = MeanEmbeddingVectorizer(w2v).transform(X)
        #print(X[labelled_set], X[labelled_set].shape)
        #sys.exit()
        
        #print(X, type(X), X.shape)
        X_train, X_test = X[labelled_set], X[unlabelled_set]
        y_train = y[labelled_set]
        return X_train, y_train, X_test