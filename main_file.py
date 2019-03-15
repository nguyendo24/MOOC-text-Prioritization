# -*- coding: utf-8 -*-
#
# author: amanul haque
#

import numpy as np
import pandas as pd
import sys
from word_to_vec_vectorizer import vectorization
from data_preprocessing_1 import data_preprocessing_1
from data_preprocessing_2 import data_preprocessing_2
from data_preprocessing_3 import data_preprocessing_3
from semi_supervised_classification import semi_supervised_classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from supervised_models import classification
import math

class main_file:
    
    def __init__(self):
        
        self.input_file_path = "output_file.csv"
        self.output_file_path = "output_file_2.csv"
        
    def get_input_text_and_label(self, df):
        #print(data['text'])
        X = np.array(df['Text'])
        y = np.array(df['Label'])
        
        return X, y
    
    def get_test_train_split(self, X, y):
        unlabelled_set = []
        labelled_set = []
        labels = []
        index = 0
        for label in y:
            if label == "NA" or math.isnan(label):
                unlabelled_set.append(index)
                labels.append(-1)
            else:
                labelled_set.append(index)
                labels.append(label)
            index+=1
        
        unlabelled_set = np.array(unlabelled_set)
        labelled_set = np.array(labelled_set)  
        labels = np.array(labels)
        #print("unlabelled_set ", unlabelled_set, unlabelled_set.shape)
        #print("labelled_set :", labelled_set, labelled_set.shape)
        #print("lables: ", labels, labels.shape)
        
        return labels, labelled_set, unlabelled_set

    
    def get_vectorized_data(self, X, y, labelled_set, unlabelled_set):
        vectorizor = vectorization()
        #X, y = vectorizor.get_data(filepath)
        X_train, y_train, X_test = vectorizor.word2vec_vectorization(X, y, labelled_set, unlabelled_set)
        return  X_train, y_train, X_test
    
    def update_dataframe(self, df, X):
        
        i = 0
        column_headers = df.columns.values
        df_new = pd.DataFrame(columns=column_headers)
        
        for index, row in df.iterrows():
            row['Text'] = X[i]
            df_new.loc[i] = row
            i += 1
        return df_new
    
    def remove_nan(self, X):
        new_X = []
        for x in X:
            if x == 'nan':
                new_X.append("empty_string")
            else:
                new_X.append(x)
                
                
        return np.array(new_X)
        
    def classification_rep(self, X_test, y_true, clf):
        y_pred = clf.predict(X_test)
        target_names = ['Non_Urgent', 'Urgent']
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def confusion_mat(self, X_test, y_true, clf):
        y_pred = clf.predict(X_test)
        pred_conf = clf.decision_function(X_test)
        print("Final labelling")
        for y_p, y_t, conf in zip(y_pred, y_true, pred_conf):
            print(y_p, "\t", y_t, "\t", conf)
        return confusion_matrix(y_true, y_pred, labels=[0, 1])  

    def create_new_dataset(self, df):
        count_one = 0
        count_zero = 0
        i=0
        x = 10
        column_headers = df.columns.values
        df_new = pd.DataFrame(columns=column_headers)
        for index, row in df.iterrows():
            if(index > 300):
                if(row['Label'] == 1 and count_one<x):
                    count_one+=1
                    
                elif(row['Label'] == 0 and count_zero<x):
                    count_zero+=1
                if(i<200):
                    df_new.loc[i] = row
                    i += 1
        print("count_one : ", count_one)
        print("count_zero : ", count_zero)
        df_new.to_csv('sample_data_3.csv')
        return df_new
    
    def show_topK(self, classifier, vectorizer, categories, K=10):
        feature_names = np.asarray(vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            topK = np.argsort(classifier.coef_[0])[-K:]
            print("%s: %s" % (category, " ".join(feature_names[topK])))        
        
    
if __name__ == '__main__':
    
    mf = main_file()
    
    df = pd.read_csv(mf.input_file_path, sep=',')
    X, y = mf.get_input_text_and_label(df)
    #print(mf.create_new_dataset(df))
    #sys.exit()
    
    '''
    X = data_preprocessing_1().process_data(X)
    X = data_preprocessing_2().process_data(X)    
    X = data_preprocessing_3().preprocess_text(X)
    
    X = mf.remove_nan(X)
    
    df = mf.update_dataframe(df, X)
    df.to_csv(mf.output_file_path)
    
    sys.exit()
    '''
    
    y, labelled_set, unlabelled_set = mf.get_test_train_split(X, y)
    
    '''
    print(y, y.shape)
    print(labelled_set, labelled_set.shape)
    print(unlabelled_set, unlabelled_set.shape)
    
    sys.exit()
    '''
    
    print("X shape ", X.shape)
    #X_train, y_train, X_test = mf.get_vectorized_data(X, y, labelled_set, unlabelled_set)
    X_train, y_train, X_test, vectorizer = vectorization().tfidf_vectorization(X, y, labelled_set, unlabelled_set)
    '''
    print(y_train, y_train.shape)
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    sys.exit()
    '''
    print("Vectorized X train shape ", X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    #sys.exit()
    
    '''
    cl = classification()
    predicted_labels, prediction_confidence, clf = cl.logistic_regression(X_train, y_train, X_test)
    print("final_labels :", predicted_labels, predicted_labels.shape)
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    sys.exit()
    '''
    
    sample_rate=0.2
    final_labels, clf = semi_supervised_classification().pseudo_labelling(y, X_train, y_train, X_test, labelled_set, unlabelled_set, sample_rate)
    
    '''
    print("y : ", y)
    print("labelled_set : ", labelled_set)
    print("unlablled_set : ", unlabelled_set)
    '''
    
   
    print("final_labels :", final_labels, final_labels.shape)
    unique, counts = np.unique(final_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    print(mf.classification_rep(X_train, y_train, clf))
    print(mf.confusion_mat(X_train, y_train, clf))
    
    mf.show_topK(clf, vectorizer, ['0', '1'], K=10)
    
    #df = mf.update_dataframe(df, X)
    #df.to_csv(mf.output_file_path)
    
    
        
    