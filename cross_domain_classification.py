# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code trains on one dataset and tests on another.
#                   The following code does it for MOOC datasets

import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
    
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from Semi_EM_NB import Semi_EM_MultinomialNB
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier



class cross_domain_classification:
    
    def __init__(self):
        
        self.input_file1 = 'Data/BDE2013_processed.csv'
        self.input_file2 = 'Data/BDE2015_processed.csv'
        self.input_file3 = "Data/processed_final_dataset.csv"
        
        
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
        custom_tags = {'q_mark':0, 'what':0, 'where':0, 'how':0, 'why':0, 'which':0, 'where':0}
        for tag in custom_tags:
            if tag in word_freq:
                custom_tags[tag] = 1
            else:
                custom_tags[tag] = 0
        #print(Counter(custom_tags))
        return Counter(custom_tags)
        
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


       
        
if __name__ == '__main__':
    
    cdc = cross_domain_classification()
    data1 = pd.read_csv(cdc.input_file1)
    data2 = pd.read_csv(cdc.input_file2)
    
    X_train, y_train = data1['post_text'], data1['category']
    X_test, y_test = data2['post_text'], data2['category']
       
    classifiers = [          
            ('logistic regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
            ("Decision Tree ", DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)),
            ("Random Forest", RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)),
            ("Extra Tree Classifier ", ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)),
            ("Linear SVM ", LinearSVC(random_state=0, tol=1e-5)),
            ("Gaussian NB ", GaussianNB()),
            ("Multinomial NB ", MultinomialNB()),
            ("Multi-Layer Perceptron", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 20), random_state=1)),
            ("K-Nearest Neighbors ", KNeighborsClassifier(n_neighbors=50))
            ]
    
    classifiers = [          
            #('logistic regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
            #("Decision Tree ", DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)),
            #("Random Forest", RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)),
            #("K-Nearest Neighbors ", KNeighborsClassifier(n_neighbors=50)),
            ("Linear SVM ", LinearSVC(random_state=0, tol=1e-5)),
            #("Gaussian NB ", GaussianNB())            
            ]
    
    n_grams = (1,3)
    results = pd.DataFrame()
    
    for clf_name, clf in classifiers:
        print("Classifier : ", clf_name)
        ppl = Pipeline([
                ('feats', FeatureUnion([
                  ('ngram', TfidfVectorizer(ngram_range=n_grams, use_idf=True, smooth_idf=True, norm='l2')),
                  ('custom_features', Custom_features())
                  ])),
                  #('to_dense', DenseTransformer()),
                  ('clf',  clf)
            ])
        
        ppl.fit(X_train, y_train)
        y_pred = ppl.predict(X_test)
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
        
        
        confusion_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = confusion_mat.ravel()
        
        
        print(" accuracy = ", (tp + tn)/(tp + tn + fp + fn))   
        u_precision = tp/(tp + fp)
        u_recall = tp/(tp + fn)
        u_f1_score = 2 * u_precision * u_recall / (u_precision + u_recall)
        
        non_u_precision = tn/(tn + fn)
        non_u_recall = tn/(tn + fp)
        non_u_f1_score = 2 * non_u_precision * non_u_recall / (non_u_precision + non_u_recall)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        d = [[clf_name, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy]]
        newDF = pd.DataFrame(data = d, columns = ['model','u_precision', 'u_recall', 'u_f1_score', 'non_u_precision', 'non_u_recall', 'non_u_f1_score', 'accuracy'])
        print(newDF)
        results = results.append(newDF)
        
    results.to_csv("Results/cross_domain_results.csv")