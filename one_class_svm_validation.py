# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code tests the performance of a one-class svm over the 150 labelled ub=nstances in JAVA2015 dataset.


import pandas as pd
import numpy as np
import math
import pickle
import nltk
import sys
from main_file import main_file
from feature_selection import feature_selection
from sklearn import tree
from sklearn.metrics import accuracy_score
from supervised_models import classification


from supervised_models import classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from validation import validation
from validation_2 import validation_2
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from Semi_EM_NB import Semi_EM_MultinomialNB
from textblob import TextBlob
from textblob.classifiers import PositiveNaiveBayesClassifier
from scipy import sparse
from sklearn.semi_supervised import LabelPropagation
from compare_labels_probabilities import compare_labels_probabilities
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelSpreading

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from sklearn.pipeline import Pipeline, FeatureUnion

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
                
class Custom_features(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, normalize=True):
        self.normalize=normalize

    #Transformer for custom text features
    def custom_feat(self, sentence):  
        tokenized_word = nltk.word_tokenize(sentence)
        word_freq = Counter(tokenized_word)
        custom_tags = {'q_mark':0, 'what':0, 'where':0, 'how':0, 'why':0, 'which':0, 'who':0}
        for tag in custom_tags:
            if tag in word_freq:
                custom_tags[tag] = 1
            else:
                custom_tags[tag] = 0
        '''
        if 'q_mark' in sentence:
            dict = {'q_mark':1.0, 'non_qmark':0.0}
        else:
            dict = {'q_mark':0.0, 'non_qmark':1.0}
        '''
        #print(Counter(custom_tags))
        return Counter(custom_tags)
        #return Counter(tag for word,tag in nltk.pos_tag(self.tokenizer(sentence)))

    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        X = pd.Series(X)        
        X_tagged = X.apply(self.custom_feat).apply(pd.Series).fillna(0)
        X_tagged['n_tokens'] = X_tagged.apply(sum, axis=1)
        #print("Xxxx ", X_tagged)
        if self.normalize:
            X_tagged = X_tagged.divide(X_tagged['n_tokens'], axis=0).fillna(0)
        #print("X tagged ", X_tagged)
        return X_tagged


class Custom_features_2(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, key):        
        self.key = key
        
    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        return np.transpose(np.matrix(X[self.key]))
    
    
class Custom_features_3(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, key):        
        self.key = key
        
    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        return X[self.key]  
    
class one_class_svm_validation:
    
    def __init__(self):
        
        #self.input_file = 'Data/BDE2013_processed.csv'
        #self.input_file_2 = 'Data/BDE2015_processed_old.csv'
        self.input_file = "Data/JAVA2015_lematized_stemmed.csv"
        #self.input_file_3 = "Data/JAVA2015_stemmed.csv"
        
    def get_test_train_split(self, y, response_needed = False):
        unlabelled_set = []
        labelled_set = []
        labels = []
        index = 0
        for label in y:
            if not response_needed and (label == 1 or label == 0):
                labelled_set.append(index)
                labels.append(label)
            elif response_needed and label == 1:
                labelled_set.append(index)
                labels.append(label)
            else:
                unlabelled_set.append(index)
                labels.append(-1)
            index+=1
        
        unlabelled_set = np.array(unlabelled_set)
        labelled_set = np.array(labelled_set)  
        labels = np.array(labels)
        
        return labels, labelled_set, unlabelled_set
    
        
    def basic_model(self, X, y, n_gram):
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        final_confusion_matrix = [[0,0],[0,0]] 
            
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("X train ", X_train.shape)
            print("X test ", X_test.shape)
                       
            train_index_orig = labelled_set[train_index]
            test_index_orig = labelled_set[test_index]
            
            print("X_train shape ", X_train.shape)
            
            print("train_index_orig shape ", train_index_orig.shape)
            print("test_index_orig shape ", test_index_orig.shape)
            
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            
            ppl = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        #('Custom_features_ppl', Pipeline([
                        #    ('selector', Custom_features()),
                        #])),            
                        
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('tfidf',  TfidfVectorizer(ngram_range = n_gram, use_idf=True, smooth_idf=True, norm='l2')),
                        ])),
            
                    ],            
                    # weight components in FeatureUnion
                    transformer_weights={
                        #'Custom_features_ppl': 1.0,
                        'text_ppl': 1.0,
                    },
                )),
                        #('to_dense', DenseTransformer()),
                        ('clf', clf)
            ])
                        
            ppl.fit(X_train)
            y_pred = ppl.predict(X_test)
            y_pred[y_pred == -1] = 0
            confusion_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
            print(confusion_mat)
            tn, fp, fn, tp = confusion_mat.ravel()
            print(tn, fp, fn, tp)
            final_confusion_matrix[0][0] += tn
            final_confusion_matrix[0][1] += fp
            final_confusion_matrix[1][0] += fn
            final_confusion_matrix[1][1] += tp

        
        tn, fp, fn, tp = np.array(final_confusion_matrix).ravel()
        
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        
        return np.array(final_confusion_matrix), accuracy


if __name__ == '__main__':
    
    ocm = one_class_svm_validation()    
    data = pd.read_csv(ocm.input_file) 
    
    n_gram = (1,4)
    X = np.array(data['Text'])
    y = np.array(data['Response_needed'])
    print(X.shape, "\t", y.shape)
    
    y, labelled_set, unlabelled_set = ocm.get_test_train_split(y, response_needed = True)
    X = X[labelled_set]
    y = y[labelled_set]
    
    conf_matrix, accuracy = ocm.basic_model(X, y, n_gram)
    
    print("Final Confusion Matrix : ", conf_matrix)
    print("Accuracy = ", accuracy)