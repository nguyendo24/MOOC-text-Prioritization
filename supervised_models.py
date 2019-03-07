# -*- coding: utf-8 -*-
#
# author: amanul haque
#
from word_to_vec_vectorizer import vectorization
from sklearn import svm
from sklearn.svm import LinearSVC

class classification:
    
    def __init__(self):
        self.a = 0
            
    def classifier_1(self, X_train, y_train, X_test, labelled_set, unlabelled_set):
        clf = svm.SVC(kernel = 'linear', C=1.0, probability = True)
        clf.fit(X_train, y_train)  
        predicted_labels = clf.predict(X_test)
        predicted_probabilities = clf.predict_proba(X_test)
        return predicted_labels, predicted_probabilities
        
    def classifier_2(self, X_train, y_train, X_test, labelled_set, unlabelled_set):
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(X_train, y_train)
        predicted_labels = clf.predict(X_test)
        prediction_confidence = clf.decision_function(X_test)
        return predicted_labels, prediction_confidence, clf
       
# =============================================================================
# cl = classification()
# X_train, y_train, X_test, labelled_set, unlabelled_set = cl.get_vectorized_data()
# predicted_labels, prediction_confidence = cl.classifier_2(X_train, y_train, X_test, labelled_set, unlabelled_set)
# print(predicted_labels, predicted_labels.shape)
# print(prediction_confidence, prediction_confidence.shape)
# =============================================================================
