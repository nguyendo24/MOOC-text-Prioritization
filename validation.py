# -*- coding: utf-8 -*-
#
# author: amanul haque
#
from sklearn.model_selection import StratifiedKFold
from semi_supervised_classification import semi_supervised_classification

from supervised_models import classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
import numpy as np

class validation:
    
    def __init__(self):
        self.x = 0
        self.cl = classification()
        
    def classification_rep(self, X_test, y_true, clf):
        y_pred = clf.predict(X_test)
        target_names = ['Non_Urgent', 'Urgent']
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def confusion_mat(self, X_test, y_true, clf):
        y_pred = clf.predict(X_test)
        #pred_conf = clf.decision_function(X_test)
        #print("Final labelling")
        #for y_p, y_t, conf in zip(y_pred, y_true, pred_conf):
        #    print(y_p, "\t", y_t, "\t", conf)
        return confusion_matrix(y_true, y_pred, labels=[0, 1])  

        
    def stratified_cross_validation(self, X_labelled, y_labelled, X_unlabelled, y, labelled_set, unlabelled_set):
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        labels = np.copy(y)
        final_confusion_matrix = [[0,0],[0,0]] 
        for train_index, test_index in skf.split(X_labelled, y_labelled):
            y = np.copy(labels)
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_labelled[train_index], X_labelled[test_index]
            y_train, y_test = y_labelled[train_index], y_labelled[test_index]
            labelled_set = train_index
            print("y shape before ", y.shape)
            print("Y before ", y)
            y = np.delete(y, test_index)
            print("y shape after ", y.shape)
            print("y after ", y)
            sample_rate=0.2
            print("Y before ", labels)
            print("X train ", X_train.shape)
            print("X test ", X_test.shape)
            #final_labels, clf = semi_supervised_classification().pseudo_labelling(y, X_train, y_train, X_unlabelled, labelled_set, unlabelled_set, sample_rate)
            final_labels, clf = self.cl.label_propagation(X_train, y, X_unlabelled)
            print("Y after ", labels)
            pred_labels = clf.predict(X_test)
            print("pred_labels :", pred_labels, "\tReal labels: ", y_test)
            print(self.classification_rep(X_train, y_train, clf))
            confusion_matrix = self.confusion_mat(X_test, y_test, clf)
            print(confusion_matrix)
            tn, fp, fn, tp = confusion_matrix.ravel()
            print(tn, fp, fn, tp)
            final_confusion_matrix[0][0] += tn
            final_confusion_matrix[0][1] += fp
            final_confusion_matrix[1][0] += fn
            final_confusion_matrix[1][1] += tp

            print("Final confiusion matrix ", final_confusion_matrix)  
        
        tp, fp, fn, tp = final_confusion_matrix[0][0], final_confusion_matrix[0][1], final_confusion_matrix[1][0], final_confusion_matrix[1][1]
        overall_precision = tp/(tp + fp)
        overall_recall = tp/(tp + fn)
        overall_accuracy = (tp + tn)/(tp + tn + fp + fn)
        overall_f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        return np.array(final_confusion_matrix), overall_precision, overall_recall, overall_accuracy, overall_f1_score
            
        