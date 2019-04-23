# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code contained methods fot supervised classification algorithms


from word_to_vec_vectorizer import vectorization
from sklearn import svm
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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
from scipy import sparse
from sklearn.semi_supervised import LabelSpreading
from transductive_SVM import transductive_SVM

class classification:
    
    def __init__(self):
        self.a = 0
            
    def classifier_1(self, X_train, y_train, X_test, labelled_set, unlabelled_set):
        clf = svm.SVC(kernel = 'linear', C=1.0, probability = True)
        clf.fit(X_train, y_train)  
        predicted_labels = clf.predict(X_test)
        predicted_probabilities = clf.predict_proba(X_test)
        return predicted_labels, predicted_probabilities
        
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
       
    #def calibration_isotonic(self, X_train, y_train, X_test, clf)
    
    def expectation_maximization(self,  X_train, y_train, X_test):
        
        nb_clf = MultinomialNB(alpha=1e-2)
        #print("XX ", X_train.shape, "\tYY ", y_train.shape)
        nb_clf.fit(X_train, y_train)
        #print("XXXXXX ", X_test.shape)
        #print(X_test)
        
        # Train Naive Bayes classifier (imported) 
        # using both labeled and unlabeled data set
        em_nb_clf = Semi_EM_MultinomialNB(alpha=1e-2) # semi supervised EM based Naive Bayes classifier
        #X_test = np.asmatrix(X_test)
        #X_test = sparse.csr_matrix(X_test)
        #print("TYPE : ", type(X_test))
        em_nb_clf.fit(X_train, y_train, X_test)
        # em_nb_clf.fit_with_clustering(X_l, y_l, X_u)
        # em_nb_clf.partial_fit(X_l, y_l, X_u)
        
        '''
        # Evaluate original NB classifier using test data set
        pred = nb_clf.predict(X_test)
        print("Normal Naive Bayes")
        print(pred)
        print(metrics.classification_report(test_Xy.target, pred, target_names=['1','2']))
        # pprint(metrics.confusion_matrix(test_Xy.target, pred))
        print(metrics.accuracy_score(test_Xy.target, pred))
        '''
        
        # Evaluate semi-supervised EM NB classifier using test data set
        pred = em_nb_clf.predict(X_test)
        print("EM Semi supervised")
        #print(pred)
        #unique, counts = np.unique(pred, return_counts=True)
        #print(dict(zip(unique, counts)))
        return pred, em_nb_clf
        #sys.exit()
        #print(metrics.classification_report(test_Xy.target, pred, target_names=['1','2']))
        # pprint(metrics.confusion_matrix(test_Xy.target, pred))
        #print(metrics.accuracy_score(test_Xy.target, pred))
        
        # find the most informative features 
        #import numpy as np
        '''
        def show_topK(classifier, vectorizer, categories, K=10):
            feature_names = np.asarray(vectorizer.get_feature_names())
            for i, category in enumerate(categories):
                topK = np.argsort(classifier.coef_[i])[-K:]
                print("%s: %s" % (category, " ".join(feature_names[topK])))
                
        #show_topK(nb_clf, vectorizer, ['1','2'], K=10) # keywords for each class by original NB classifier
        
        #show_topK(em_nb_clf, vectorizer, ['1','2'], K=10) # keywords for each class by semisupervised EM NB classifier
        '''
        #print(nb_clf.class_log_prior_, em_nb_clf.clf.class_log_prior_)
        
        
    def label_propagation(self, X_train, y, X_test):
        
        clf = LabelPropagation()
        print("X_train Shape :", X_train.shape, type(X_train))
        print("X_test shape : ", X_test.shape, type(X_test))
        print("y shape : ", y.shape )
        
        X = np.concatenate((X_train.todense(), X_test.todense()),axis=0)
        print("X shape now ", X.shape)
        print("Y shape now ", y.shape)
        clf.fit(X, y)
        final_labels = clf.predict(X_test)
        label_prob = clf.predict_proba(X_test)
        print(compare_labels_probabilities().compare(label_prob, final_labels))
        return final_labels, clf
    
    
    def decision_tree(self, X_train, y_train, X_test):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        final_labels = clf.predict(X_test)
        return final_labels, clf
    
    def random_forest(self, X_train, y_train, X_test):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        final_labels = clf.predict(X_test)
        return final_labels, clf
        
    
    def label_spreading(self, X_train, y, X_test):
        clf = LabelSpreading()
        X = np.concatenate((X_train.todense(), X_test.todense()),axis=0)
        print("X shape now ", X.shape)
        print("Y shape now ", y.shape)
        clf.fit(X, y)
        final_labels = clf.predict(X_test)
        label_prob = clf.predict_proba(X_test)
        print(compare_labels_probabilities().compare(label_prob, final_labels))
        return final_labels, clf
    
    def transductive_svm(self, X_train, y, X_test):
        clf = transductive_SVM()
        X = np.concatenate((X_train.todense(), X_test.todense()),axis=0)
        print("X shape now ", X.shape)
        print("Y shape now ", y.shape)
        y = y.astype(int)
        print(y)
        clf.fit(X, y)
        final_labels = clf.predict(X_test)
        #label_prob = clf.predict_proba(X_test)
        #print(compare_labels_probabilities().compare(label_prob, final_labels))
        return final_labels, clf
        
    
    def one_class_svm(self, X_train, y_train, X_test):
        clf = svm.OneClassSVM(nu = 0.5, kernel = 'rbf', gamma = 0.1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        unique, counts = np.unique(y_pred, return_counts=True)
        print(dict(zip(unique, counts)))
        return y_pred, clf
        
# =============================================================================
# cl = classification()
# X_train, y_train, X_test, labelled_set, unlabelled_set = cl.get_vectorized_data()
# predicted_labels, prediction_confidence = cl.classifier_2(X_train, y_train, X_test, labelled_set, unlabelled_set)
# print(predicted_labels, predicted_labels.shape)
# print(prediction_confidence, prediction_confidence.shape)
# =============================================================================
