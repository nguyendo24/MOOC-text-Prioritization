# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code implements the Model 1a and Model 1b.
#                   It trains a supervised model on combined BDE2015 and BDE2013 datasets and 
#                   combines its predictions with a semi-suervised model trained on limted labeled instances from JAVA2015.


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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from feature_model import feature_model
from filtering_model import filtering_model
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

from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from sklearn.pipeline import Pipeline, FeatureUnion

class question_detector:
    
    def __init__(self):
        
        #self.input_file = 'Data/BDE2013_wo_TXSP.csv'
        #self.input_file_2 = 'Data/BDE2015_wo_TXSP.csv'
        self.input_file = 'Data/BDE2013_processed.csv'
        self.input_file_2 = 'Data/BDE2015_processed.csv'
        self.input_file_3 = "Data/JAVA2015_lematized_stemmed.csv"
        #self.input_file_3 = "Data/JAVA2015_stemmed.csv"
        
    def tfidf_vectorization(self, X):
        #vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
        vectorizer = TfidfVectorizer(ngram_range=(1,4), norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
        #print(X.shape)
        #vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(X)
        tf = X.toarray()
        transformer = TfidfTransformer()
        Y = transformer.fit_transform(tf)
        tfidf = Y.toarray()
        
        return tfidf, vectorizer
    
    def classifier_1(self, X, y):
        clf = svm.SVC(kernel = 'linear', C=1.0, probability = True)
        clf.fit(X, y)  
        return clf
        
    def linear_svc(self, X_train, y_train, X_test):
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(X_train, y_train)
        predicted_labels = clf.predict(X_test)
        prediction_confidence = clf.decision_function(X_test)
        return predicted_labels, prediction_confidence, clf
    
    def radial_svc(self, X_train, y_train, X_test):
        clf = SVC(gamma='auto', kernel='rbf')
        clf.fit(X_train, y_train)
        predicted_labels = clf.predict(X_test)
        prediction_confidence = clf.decision_function(X_test)
        return predicted_labels, prediction_confidence, clf
    
    def logistic_regression(self, X_train, y_train, X_test):
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
        predicted_labels = clf.predict(X_test)
        prediction_confidence = clf.decision_function(X_test)
        final_labels = clf.predict(X_test)
        label_prob = clf.predict_proba(X_test)
        print(compare_labels_probabilities().compare(label_prob, final_labels))
        return predicted_labels, prediction_confidence, clf
    
    def remove_nan(self, X):
        new_X = []
        for x in X:
            if x == 'nan' or math.isnan(float(x)):
                new_X.append("empty_string")
            else:
                new_X.append(x)

        return np.array(new_X)
       
    def test_model(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))
        print("Accuracy ", accuracy_score(y_test,y_pred))
            
    def show_most_informative_features(self, vectorizer, clf, n=20):
        feature_names = vectorizer.get_feature_names()
        '''
        print(len(feature_names))
        unique, counts = np.unique(clf.coef_, return_counts=True)
        dic = dict(zip(unique, counts))
        print(sorted(dic.items(), key=operator.itemgetter(0)))
        '''
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        l = []
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
            l.append(fn_2)
            
        print(l)
        
    def custom_feat(self, sentence):
        tokenized_word = nltk.word_tokenize(sentence)
        word_freq = Counter(tokenized_word)
        custom_tags = {'q_mark':0, 'what':0, 'where':0, 'how':0, 'why':0, 'which':0}
        for tag in custom_tags:
            if tag in word_freq:
                custom_tags[tag] = word_freq[tag]
                
        print(custom_tags)
        
        
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

        
if __name__ == '__main__':
    
    qd = question_detector()
    feature_sel = feature_selection()
    #sys.exit()
    
    data = pd.read_csv(qd.input_file)    
    X_train_1 = data['post_text']
    y_train_1 = data['category']
    print(X_train_1.shape, "\t", y_train_1.shape)
    
    data = pd.read_csv(qd.input_file_2)    
    X_train_2 = data['post_text']
    y_train_2 = data['category']
    print(X_train_2.shape, "\t", y_train_2.shape)
    
    
    uniques, count = np.unique(y_train_2, return_counts = True)
    print(dict(zip(uniques,count)))
    sys.exit()
    
    X = np.concatenate((X_train_1, X_train_2),axis=0)
    y = np.concatenate((y_train_1, y_train_2),axis=0)
    
    train_index = np.arange(len(X))
    #print(train_index)
    
    mf = main_file()    
    df = pd.read_csv(qd.input_file_3)   
    X_test, y_test = np.array(df['Text']), np.array(df['Urgency'])    
    
    y_test, labelled_set, unlabelled_set = mf.get_test_train_split(X_test, y_test)
    
    test_index = np.arange(150) + len(train_index)
    response_label = df['Response_needed']
    response_label = np.array(response_label[labelled_set])
    
    classifiers = [          
            ('logistic regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
            ("Decision Tree ", DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)),
            ("Random Forest", RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)),
            ("Linear SVM ", LinearSVC(random_state=0, tol=1e-5)),
            ("K-Nearest Neighbors ", KNeighborsClassifier(n_neighbors=50))
            ]
    
    results = pd.DataFrame()
    index = 0
    ngrams = (1,2)
    semi_clf = 'EM'
    
    for clf_name, clf in classifiers:
        print(clf_name)
        
        ppl = Pipeline([
                ('feats', FeatureUnion([
                  ('ngram', TfidfVectorizer(ngram_range = (1,3), use_idf=True, smooth_idf=True, norm='l2')),
                  ('custom_features', Custom_features())
                  ])),
                  #('to_dense', DenseTransformer()),
                  ('clf', clf)
          ])
        
        cv = feature_model()           #Model 1 (Feature Model)
        cv2 = filtering_model()        #Model 2 (Filtering model)
        
        conf_matrix, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy = cv2.skfold_cv(X, y, X_test, y_test, response_label, labelled_set, unlabelled_set, ppl, df, ngrams, semi_clf)
        d = [[clf_name, u_precision, u_recall, u_f1_score, non_u_precision, non_u_recall, non_u_f1_score, accuracy]]
        newDF = pd.DataFrame(data = d, columns = ['model','u_precision', 'u_recall', 'u_f1_score', 'non_u_precision', 'non_u_recall', 'non_u_f1_score', 'accuracy'])
        print(newDF)
        results = results.append(newDF)
        
    results.to_csv("Results/results_2.csv")