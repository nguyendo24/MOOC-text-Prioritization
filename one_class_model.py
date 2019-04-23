# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code is for One-class SVM mode
#                   This model trains a partially supervised classifier on JAVA2015 dataset first to identify questions and non-questionsin the unlabelled instances
#                   then tries to fing urgenvy using the semi-supervised model on topof it.

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
    
class one_class_model:
    
    def __init__(self):
        
        #self.input_file = 'Data/BDE2013_processed.csv'
        #self.input_file_2 = 'Data/BDE2015_processed_old.csv'
        self.input_file = "Data/reduced_labelled_dataset.csv"
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
    
    def filtering_model(self, data, X, X_vec, y1, y2, labelled_set, unlabelled_set, n_gram, clf_name):
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        #vectorizer = TfidfVectorizer(ngram_range=(1,1), use_idf=True, smooth_idf=True, norm='l2')
        #X_vec = vectorizer.fit_transform(X)
        resp_label = np.copy(y1)
        urg_labels = np.copy(y2)
        final_confusion_matrix = [[0,0],[0,0]] 
        X_labelled = X[labelled_set]
        y_labelled = y2[labelled_set]
        X_unlabelled = X[unlabelled_set]
        #print(X_vec[labelled_set].shape)
        for train_index, test_index in skf.split(X_vec[labelled_set], y_labelled):
            #y = np.copy(labels)
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_labelled[train_index], X_labelled[test_index]
            y_train, y_test = y_labelled[train_index], y_labelled[test_index]
            #labelled_set = train_index
            #print("y shape before ", y.shape)
            y1 = np.delete(y1, test_index)
            y2 = np.delete(y2, test_index)
            cl = classification()
            print("y shape after ", y1.shape, y2.shape)
            #print("Y before ", labels)
            print("X train ", X_train.shape)
            print("X test ", X_test.shape)
                       
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            ppl = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        ('Custom_features_ppl', Pipeline([
                            ('selector', Custom_features()),
                        ])),            
                        
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('tfidf',  TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')),
                        ])),
            
                    ],            
                    # weight components in FeatureUnion
                    transformer_weights={
                        'Custom_features_ppl': 1.0,
                        'text_ppl': 1.0,
                    },
                )),
                        #('to_dense', DenseTransformer()),
                        ('clf', clf)
            ])
    
            ppl.fit(X_train)
            y_pred = ppl.predict(X_unlabelled)
            filtered_index_orig = unlabelled_set[np.where(y_pred==1)[0]]
            print(filtered_index_orig.shape)
            X_filtered = X[filtered_index_orig]
            y_test_new = resp_label[filtered_index_orig]
            y_ = np.concatenate((y_train, y_test_new), axis=0)
            
            print("X unlabelled shape ", X_filtered.shape)
            
            train_index_orig = labelled_set[train_index]
            test_index_orig = labelled_set[test_index]
            print("train_index_orig shape ", train_index_orig.shape)
            print("test_index_orig shape ", test_index_orig.shape)
            combined_train_index_orig = np.concatenate((train_index_orig, test_index_orig, filtered_index_orig),axis=0)
            print("combined_train_index_orig shape ", combined_train_index_orig.shape)
            
            
            train_df = data.iloc[combined_train_index_orig,:]
            
            pipeline = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        # Pipeline for pulling features from the post's subject line
                        ('deadline_ppl', Pipeline([
                            ('selector', Custom_features_2(key = 'deadline_weight')),
                        ])),
            
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('selector', Custom_features_3(key = 'Text')),
                            ('tfidf', TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')),
                        ]))
            
                    ],
            
                    # weight components in FeatureUnion
                    transformer_weights={
                        'deadline_ppl': 1.0,
                        'text_ppl': 1.0,
                    },
                )),
            ])
            
            #vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, smooth_idf=True, norm='l2')
            X_ = pipeline.fit_transform(train_df)
            print(X_.shape)
            
            X_train_vec = X_[0: train_index_orig.shape[0]]
            X_test = X_[train_index_orig.shape[0]:train_index_orig.shape[0]+test_index_orig.shape[0]] 
            X_unlabelled_vec = X_[-filtered_index_orig.shape[0]:]  
            
            print(X_train_vec.shape, X_unlabelled_vec.shape, X_test.shape)
                        
            #final_labels, clf = semi_supervised_classification().pseudo_labelling(y, X_train, y_train, X_unlabelled, labelled_set, unlabelled_set, sample_rate)
            if(clf_name == 'EM'):
                final_labels, clf = cl.expectation_maximization(X_train_vec, y_train, X_unlabelled_vec)
            elif(clf_name == 'LS'):
                final_labels, clf = cl.label_spreading(X_train_vec, y_, X_unlabelled_vec)
            elif(clf_name == 'LP'):
                final_labels, clf = cl.label_propagation(X_train_vec, y_, X_unlabelled_vec)
            #print("Y after ", labels)
            pred_labels = clf.predict(X_test)
            print("pred_labels :", pred_labels, "\tReal labels: ", y_test)
            confusion_mat = confusion_matrix(y_test, pred_labels, labels=[0, 1])
            print(confusion_mat)
            tn, fp, fn, tp = confusion_mat.ravel()
            print(tn, fp, fn, tp)
            final_confusion_matrix[0][0] += tn
            final_confusion_matrix[0][1] += fp
            final_confusion_matrix[1][0] += fn
            final_confusion_matrix[1][1] += tp

            print("Final confiusion matrix ", final_confusion_matrix)  
        
        tn, fp, fn, tp = np.array(final_confusion_matrix).ravel()
        
        u_precision = tp/(tp + fp)
        u_recall = tp/(tp + fn)
        u_f1_score = 2 * u_precision * u_recall / (u_precision + u_recall)
        
        non_u_precision = tn/(tn + fn)
        non_u_recall = tn/(tn + fp)
        non_u_f1_score = 2 * non_u_precision * non_u_recall / (non_u_precision + non_u_recall)
        
        
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        
        return np.array(final_confusion_matrix), u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy
           
    
    def feature_model(self, data, X, X_vec, y1, y2, labelled_set, unlabelled_set, n_gram, clf_name):
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        #vectorizer = TfidfVectorizer(ngram_range=(1,1), use_idf=True, smooth_idf=True, norm='l2')
        #X_vec = vectorizer.fit_transform(X)
        resp_label = np.copy(y1)
        urg_labels = np.copy(y2)
        final_confusion_matrix = [[0,0],[0,0]] 
        X_labelled = X[labelled_set]
        y_labelled = y2[labelled_set]
        X_unlabelled = X[unlabelled_set]
        data['feature_response_labels'] = -1
            
        #print(X_vec[labelled_set].shape)
        for train_index, test_index in skf.split(X_vec[labelled_set], y_labelled):
            #y = np.copy(labels)
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_labelled[train_index], X_labelled[test_index]
            y_train, y_test = y_labelled[train_index], y_labelled[test_index]
            #labelled_set = train_index
            #print("y shape before ", y.shape)
            y1 = np.delete(y1, test_index)
            y2 = np.delete(y2, test_index)
            cl = classification()
            print("y shape after ", y1.shape, y2.shape)
            #print("Y before ", labels)
            print("X train ", X_train.shape)
            print("X test ", X_test.shape)
            
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            #clf = IsolationForest(max_samples=100, random_state = np.random.RandomState(42), contamination='auto')
            ppl = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        ('Custom_features_ppl', Pipeline([
                            ('selector', Custom_features()),
                        ])),            
                        
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('tfidf',  TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')),
                        ])),
            
                    ],            
                    # weight components in FeatureUnion
                    transformer_weights={
                        'Custom_features_ppl': 1.0,
                        'text_ppl': 1.0,
                    },
                )),
                        #('to_dense', DenseTransformer()),
                        ('clf', clf)
            ])
                            

            lab = data['Response_needed']
            unique, count = np.unique(lab, return_counts = True)
            #print(dict(zip(unique,count)))
            
            ppl.fit(X_train)
            y_pred = ppl.predict(X_unlabelled)           
            
            filtered_index_orig_one = unlabelled_set[np.where(y_pred==1)[0]]            
            print(filtered_index_orig_one.shape)
            
            y_response_label = np.concatenate((filtered_index_orig_one, labelled_set), axis=0)
            print("Shapes ", y1.shape[0]+ test_index.shape[0])
            response_labels = []
            for i in range(data.shape[0]):
                if i in y_response_label:
                    response_labels.append(1)
                else:
                    response_labels.append(0)
            
            #print(response_labels)
            p = data['feature_response_labels']
            unique, count = np.unique(p, return_counts = True)
            print(dict(zip(unique,count)))
            
            response_labels = pd.Series(response_labels)
            print(response_labels.shape)
            
            #train_df_clf2.iloc[:,28] = combined_response_labels
            data = data.assign(feature_response_labels = response_labels.values)
            
            train_index_orig = labelled_set[train_index]
            test_index_orig = labelled_set[test_index]
            print("train_index_orig shape ", train_index_orig.shape)
            print("test_index_orig shape ", test_index_orig.shape)
            combined_train_index_orig = np.concatenate((train_index_orig, test_index_orig, unlabelled_set),axis=0)
            print("combined_train_index_orig shape ", combined_train_index_orig.shape)
            
            
            train_df = data.iloc[combined_train_index_orig,:]
            print(train_df.shape)
            
            pipeline = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        # Pipeline for pulling features from the post's subject line
                        ('deadline_ppl', Pipeline([
                            ('selector', Custom_features_2(key = 'deadline_weight')),
                        ])),
                        
                        ('response_label_ppl', Pipeline([
                            ('selector', Custom_features_2(key = 'feature_response_labels')),
                        ])),            
                        
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('selector', Custom_features_3(key = 'Text')),
                            ('tfidf',  TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')),
                        ])),
            
                    ],
            
                    # weight components in FeatureUnion
                    transformer_weights={
                        'deadline_ppl': 1.0,
                        'response_label_ppl':1.0,
                        'text_ppl': 1.0,
                    },
                )),
            ])
            #vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, smooth_idf=True, norm='l2')
            X_ = pipeline.fit_transform(train_df)
            print(X_.shape)
            
            X_train_vec = X_[0: train_index_orig.shape[0]]
            X_test = X_[train_index_orig.shape[0]:train_index_orig.shape[0]+test_index_orig.shape[0]] 
            X_unlabelled_vec = X_[-unlabelled_set.shape[0]:]  
            
            print(X_train_vec.shape, X_unlabelled_vec.shape, X_test.shape)
            y_ = np.concatenate((y_train, resp_label[unlabelled_set]),axis = 0)            
            #final_labels, clf = semi_supervised_classification().pseudo_labelling(y, X_train, y_train, X_unlabelled, labelled_set, unlabelled_set, sample_rate)
            if(clf_name == 'EM'):
                final_labels, clf = cl.expectation_maximization(X_train_vec, y_train, X_unlabelled_vec)
            elif(clf_name == 'LS'):
                final_labels, clf = cl.label_spreading(X_train_vec, y_, X_unlabelled_vec)
            elif(clf_name == 'LP'):
                final_labels, clf = cl.label_propagation(X_train_vec, y_, X_unlabelled_vec)
            
            #print("Y after ", labels)
            pred_labels = clf.predict(X_test)
            print("pred_labels :", pred_labels, "\tReal labels: ", y_test)
            confusion_mat = confusion_matrix(y_test, pred_labels, labels=[0, 1])
            print(confusion_mat)
            tn, fp, fn, tp = confusion_mat.ravel()
            print(tn, fp, fn, tp)
            final_confusion_matrix[0][0] += tn
            final_confusion_matrix[0][1] += fp
            final_confusion_matrix[1][0] += fn
            final_confusion_matrix[1][1] += tp

            print("Final confiusion matrix ", final_confusion_matrix)  
        
        tn, fp, fn, tp = np.array(final_confusion_matrix).ravel()
        
        u_precision = tp/(tp + fp)
        u_recall = tp/(tp + fn)
        u_f1_score = 2 * u_precision * u_recall / (u_precision + u_recall)
        
        non_u_precision = tn/(tn + fn)
        non_u_recall = tn/(tn + fp)
        non_u_f1_score = 2 * non_u_precision * non_u_recall / (non_u_precision + non_u_recall)
        
        
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        
        return np.array(final_confusion_matrix), u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy
    
        
    def basic_model(self, data, X, X_vec, labels, labelled_set, unlabelled_set, n_gram, clf_name):
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        #vectorizer = TfidfVectorizer(ngram_range=(1,1), use_idf=True, smooth_idf=True, norm='l2')
        #X_vec = vectorizer.fit_transform(X)
        resp_label = np.copy(labels)        
        final_confusion_matrix = [[0,0],[0,0]] 
        X_labelled = X[labelled_set]
        y_labelled = y2[labelled_set]
        X_unlabelled = X[unlabelled_set]
        data['feature_response_labels'] = -1
            
        #print(X_vec[labelled_set].shape)
        for train_index, test_index in skf.split(X_vec[labelled_set], y_labelled):
            y = np.copy(labels)
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_labelled[train_index], X_labelled[test_index]
            y_train, y_test = y_labelled[train_index], y_labelled[test_index]
            #labelled_set = train_index
            #print("y shape before ", y.shape)
            y = np.delete(y, test_index)
            cl = classification()
            print("y shape after ", y1.shape, y2.shape)
            #print("Y before ", labels)
            print("X train ", X_train.shape)
            print("X test ", X_test.shape)
                       
            train_index_orig = labelled_set[train_index]
            test_index_orig = labelled_set[test_index]
            print("train_index_orig shape ", train_index_orig.shape)
            print("test_index_orig shape ", test_index_orig.shape)
            
            pipeline = Pipeline([
                # Use FeatureUnion to combine the features from subject and body
                ('union', FeatureUnion(
                    transformer_list=[
            
                        # Pipeline for pulling features from the post's subject line
                        ('deadline_ppl', Pipeline([
                            ('selector', Custom_features_2(key = 'deadline_weight')),
                        ])),
                        
                        # Pipeline for standard bag-of-words model for body
                        ('text_ppl', Pipeline([
                            ('selector', Custom_features_3(key = 'Text')),
                            ('tfidf',  TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')),
                        ])),
            
                    ],
            
                    # weight components in FeatureUnion
                    transformer_weights={
                        'deadline_ppl': 1.0,
                        'text_ppl': 1.0,
                    },
                )),
            ])
            #vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, smooth_idf=True, norm='l2')
            X_ = pipeline.fit_transform(data)
            print(X_.shape)
            
            X_train_vec = X_[0: train_index_orig.shape[0]]
            X_test = X_[train_index_orig.shape[0]:train_index_orig.shape[0]+test_index_orig.shape[0]] 
            X_unlabelled_vec = X_[-unlabelled_set.shape[0]:]  
            y_ = np.concatenate((y_train, resp_label[unlabelled_set]),axis = 0)            
            
            print(X_train_vec.shape, X_unlabelled_vec.shape, X_test.shape)
                        
            #final_labels, clf = semi_supervised_classification().pseudo_labelling(y, X_train, y_train, X_unlabelled, labelled_set, unlabelled_set, sample_rate)
            if(clf_name == 'EM'):
                final_labels, clf = cl.expectation_maximization(X_train_vec, y_train, X_unlabelled_vec)
            elif(clf_name == 'LS'):
                final_labels, clf = cl.label_spreading(X_train_vec, y_, X_unlabelled_vec)
            elif(clf_name == 'LP'):
                final_labels, clf = cl.label_propagation(X_train_vec, y_, X_unlabelled_vec)
            
            #print("Y after ", labels)
            pred_labels = clf.predict(X_test)
            print("pred_labels :", pred_labels, "\tReal labels: ", y_test)
            confusion_mat = confusion_matrix(y_test, pred_labels, labels=[0, 1])
            print(confusion_mat)
            tn, fp, fn, tp = confusion_mat.ravel()
            print(tn, fp, fn, tp)
            final_confusion_matrix[0][0] += tn
            final_confusion_matrix[0][1] += fp
            final_confusion_matrix[1][0] += fn
            final_confusion_matrix[1][1] += tp

            print("Final confiusion matrix ", final_confusion_matrix)  
        
        tn, fp, fn, tp = np.array(final_confusion_matrix).ravel()
        
        u_precision = tp/(tp + fp)
        u_recall = tp/(tp + fn)
        u_f1_score = 2 * u_precision * u_recall / (u_precision + u_recall)
        
        non_u_precision = tn/(tn + fn)
        non_u_recall = tn/(tn + fp)
        non_u_f1_score = 2 * non_u_precision * non_u_recall / (non_u_precision + non_u_recall)
        
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        
        return np.array(final_confusion_matrix), u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy


if __name__ == '__main__':
    
    ocm = one_class_model()    
    data = pd.read_csv(ocm.input_file) 
    
    n_gram = [1,3]
    clf_name = 'EM'
    
    X = np.array(data['Text'])
    y1 = np.array(data['Response_needed'])
    y2 = np.array(data['Urgency'])
    print(X.shape, "\t", y1.shape)
    
    y1, labelled_set, unlabelled_set = ocm.get_test_train_split(y1, response_needed = True)
    y2[unlabelled_set] = -1
        
    vectorizer = TfidfVectorizer(ngram_range=(n_gram[0], n_gram[1]), use_idf=True, smooth_idf=True, norm='l2')
    X_vec = vectorizer.fit_transform(X)
    
    #Uncomment follwoing line of code for Model 2 (Feature model)
    #conf_matrix, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy = ocm.feature_model(data, X, X_vec, y1, y2, labelled_set, unlabelled_set, n_gram, clf_name)
    
    #Uncomment following line of code for Model 1 (Filtering model)
    conf_matrix, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy = ocm.filtering_model(data, X, X_vec, y1, y2, labelled_set, unlabelled_set, n_gram, clf_name)
    
    #Uncomment following line of code for basic (semi-supervied alone) Model
    #conf_matrix, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy = ocm.basic_model(data, X, X_vec, y2, labelled_set, unlabelled_set, n_gram, clf_name)
    
    
    d = [[u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy]]
    newDF = pd.DataFrame(data = d, columns = ['u_precision', 'u_recall', 'u_f1_score', 'non_u_precision', 'non_u_recall', 'non_u_f1_score', 'accuracy']) #creates a new dataframe that's empty
    #newDF['u_precision'], newDF['u_recall'], newDF['u_f1_score'] = u_precision, u_recall, u_f1_score
    #newDF['non_u_precision'], newDF['non_u_recall'], newDF['non_u_f1_score'] = non_u_precision, non_u_recall, non_u_f1_score
    #newDF['accuracy'] = accuracy
    print(newDF)
    newDF.to_csv("Results/results_one_class.csv")
        

    
    
    #print(ocm.filtering_model(data, X, X_vec, y1, y2, labelled_set, unlabelled_set, n_gram))
    
    #print(ocm.basic_model(data, X, X_vec, y2, labelled_set, unlabelled_set, n_gram))