# -*- coding: utf-8 -*-
#
## author: amanul haque
#

from word_to_vec_vectorizer import vectorization
from supervised_models import classification
import numpy as np
import pandas as pd
import math
import random
import sys

class semi_supervised_classification:
    
    def __init__(self):
        self.sample_rate = 0.2    
        self.input_file_path = ""
        self.output_file_path = ""
    
    def get_vectorized_data(self, filepath):
        vectorizor = vectorization()
        X, y = vectorizor.get_data(filepath)
        X_train, y_train, X_test, labelled_set, unlabelled_set = vectorizor.vectorize_text(X, y)
        return  X_train, y_train, X_test, labelled_set, unlabelled_set
    
    def pseudo_labelling(self, final_y, X_train, y_train, X_test, labelled_set, unlabelled_set, sample_rate):
    #def pseudo_labelling(self, X, y, X_train, y_train, X_test, X_orig):
        
        if(-1 not in final_y):
            return final_y 
        
        num_of_samples = math.ceil(len(X_train) * self.sample_rate)
        print("num_of_samples : ", num_of_samples)        
        print("Y Lables: ", final_y, final_y.shape)
        #print("X_train ", X_train, X_train.shape)
        #print("y_train ", y_train, y_train.shape)
        #print("x_test ", X_test, X_test.shape)
        print("labelled set : ", labelled_set, labelled_set.shape)
        print("unlabelled set : ", unlabelled_set, unlabelled_set.shape)
        
        cl = classification()
        predicted_labels, prediction_confidence = cl.classifier_2(X_train, y_train, X_test, labelled_set, unlabelled_set)
        print(predicted_labels, predicted_labels.shape)
        print(prediction_confidence, prediction_confidence.shape)
        sorted_indices = np.argsort(np.absolute(prediction_confidence))
        print("Sorted indices: ", sorted_indices)
        
        print("prediction confidence ",prediction_confidence[sorted_indices[-num_of_samples:]])
        
        pseudo_labelled_indices = sorted_indices[-num_of_samples:]
        print("pseudo_labelled_indices :", pseudo_labelled_indices)
        
        new_train_X = []
        new_train_y = []
        #unlabelled_indices = unlabelled_set.copy()
        for ind in pseudo_labelled_indices:
            print("Index ", ind)
            print("unlabelled_indices :", unlabelled_set)
            delete_orig_index = unlabelled_set[ind]
            print("Delete index : ", delete_orig_index)
            if(final_y[delete_orig_index] == -1):
                final_y[delete_orig_index] = predicted_labels[ind]
                new_train_y.append(predicted_labels[ind])
                new_train_X.append(X_test[ind])
                labelled_set = np.append(labelled_set, delete_orig_index)
                #unlabelled_set = np.delete(unlabelled_set, ind, axis = 0)
            else:
                print("Value already been updated : ", delete_orig_index, final_y[delete_orig_index])
                sys.exit()
        unlabelled_set = np.delete(unlabelled_set, pseudo_labelled_indices, axis = 0)    
        new_train_X = np.array(new_train_X)
        new_train_y = np.array(new_train_y)
        #print("New train X : ", new_train_X, new_train_X.shape)
        #print("New train Y : ", new_train_y, new_train_y.shape)
        X_train = np.concatenate((X_train, new_train_X), axis = 0)
        y_train = np.concatenate((y_train, new_train_y), axis = 0)
        X_test = np.delete(X_test, pseudo_labelled_indices, axis=0)
        print()
        return self.pseudo_labelling(final_y, X_train, y_train, X_test, labelled_set, unlabelled_set, sample_rate)    
    
    def write_to_csv(self, X, y, filepath):
        
        df = np.array([X, y]).T
        df = pd.DataFrame(df, columns = ['comment', 'label'])
        df.to_excel(filepath)
        print(df)
        
        
if __name__ == '__main__':
    
    ssc = semi_supervised_classification()
    X_orig, y = vectorization().get_data(ssc.input_file_path)
    X_train, y_train, X_test, labelled_set, unlabelled_set = ssc.get_vectorized_data(ssc.input_file_path)
    X = np.concatenate((X_train, X_test),axis = 0)
    sample_rate = 0.2
    #print("X is ", X, X.shape)
    final_y = np.append(y_train, np.array([-1 for x in range(len(X_test))]), axis = 0)
    print("Initial labels : ", final_y, final_y.shape)
    #final_y = ssc.pseudo_labelling(X, final_y, X_train, y_train, X_test, X_orig)
    final_y = ssc.pseudo_labelling(final_y, X_train, y_train, X_test, labelled_set, unlabelled_set, sample_rate)
    ssc.write_to_csv(X_orig, final_y, ssc.output_file_path)
    #print(X_test, X_test.shape)
