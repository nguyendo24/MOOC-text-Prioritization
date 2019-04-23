# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code does basic data-preprocessing like removing the unicodes, scraping the HTML tags etc.
#                   This is required becuase the initial generated dataset has HTML tags and other encodings

import numpy as np
import random as rnd
import nltk as nk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from pprint import pprint
import sys
import pandas as pd


from Semi_EM_NB import Semi_EM_MultinomialNB

df = pd.read_excel('sample_data.xlsx')
train_Xy = df.iloc[0:10,]
test_Xy = df.iloc[10:20,]

# Convert all text data into tf-idf vectors 
vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
# vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(train_Xy.data)
test_vec = vectorizer.transform(test_Xy.data)

# Divide train data set into labeled and unlabeled data sets
n_train_data = train_vec.shape[0]
split_ratio = 0.4 # labeled vs unlabeled
#X_l, X_u, y_l, y_u = train_test_split(train_vec, train_Xy.target, train_size=split_ratio, stratify=train_Xy.target)
X_l, X_u, y_l, y_u = train_vec[0:4], train_vec[4:10], df.iloc[0:4].target, df.iloc[4:10].target
'''
print(X_l, "\n")
print(X_u, "\n")
print(y_l, "\n")
print(y_u, "\n")
'''
#sys.exit()

# Train Naive Bayes classifier (imported) 
# using labeled data set only
nb_clf = MultinomialNB(alpha=1e-2)
nb_clf.fit(X_l, y_l)

# Train Naive Bayes classifier (imported) 
# using both labeled and unlabeled data set
em_nb_clf = Semi_EM_MultinomialNB(alpha=1e-2) # semi supervised EM based Naive Bayes classifier
em_nb_clf.fit(X_l, y_l, X_u)
# em_nb_clf.fit_with_clustering(X_l, y_l, X_u)
# em_nb_clf.partial_fit(X_l, y_l, X_u)

# Evaluate original NB classifier using test data set
pred = nb_clf.predict(test_vec)
print("Normal Naive Bayes")
print(pred)
print(metrics.classification_report(test_Xy.target, pred, target_names=['1','2']))
# pprint(metrics.confusion_matrix(test_Xy.target, pred))
print(metrics.accuracy_score(test_Xy.target, pred))


# Evaluate semi-supervised EM NB classifier using test data set
pred = em_nb_clf.predict(test_vec)
print("EM Semi supervised")
print(pred)
print(metrics.classification_report(test_Xy.target, pred, target_names=['1','2']))
# pprint(metrics.confusion_matrix(test_Xy.target, pred))
print(metrics.accuracy_score(test_Xy.target, pred))

# find the most informative features 
import numpy as np
def show_topK(classifier, vectorizer, categories, K=10):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topK = np.argsort(classifier.coef_[i])[-K:]
        print("%s: %s" % (category, " ".join(feature_names[topK])))
        
#show_topK(nb_clf, vectorizer, ['1','2'], K=10) # keywords for each class by original NB classifier

#show_topK(em_nb_clf, vectorizer, ['1','2'], K=10) # keywords for each class by semisupervised EM NB classifier

print(nb_clf.class_log_prior_, em_nb_clf.clf.class_log_prior_)

