# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code is for word feature extraction

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import nltk
import sys
import operator

class word_feature_extraction:
    
    def __init__(self):
        
        self.input_file_path = "output_file_lematized.csv"
        self.output_file_path = "output_file_2.csv"
   
    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
     
    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        
        #use only topn items from vector
        sorted_items = sorted_items[:topn]
     
        score_vals = []
        feature_vals = []
        
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            
            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
     
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results
    
    def most_common_word_features(self, X, num_of_features):
        features = {}
        for cmt in X:
            tokenized_words = nltk.word_tokenize(str(cmt))
            for word in tokenized_words:
                if(word not in features.keys()):
                    features[word] = 1
                else:
                    features[word] += 1
                    
        sorted_dict = sorted(features.items(), key=operator.itemgetter(1))
        
        return features, sorted_dict[-num_of_features:]
            
      
    def print_top10(self, vectorizer, clf, class_labels):
        """Prints features with the highest coefficient values, per class"""
        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(clf.coef_[0])[-10:]
            print("%s: %s" % (class_label,
                  " ".join(feature_names[j] for j in top10)))
            
    def show_most_informative_features(self, vectorizer, clf, n=20):
        #print("Coeff ", clf.coef_.shape)
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
                
         
    
    
if __name__ == '__main__':
    
    mf = main_file()
    wf = word_feature_extraction()
    
    df = pd.read_csv(mf.input_file_path, sep=',')
    X, y = mf.get_input_text_and_label(df)
    
    y, labelled_set, unlabelled_set = mf.get_test_train_split(X, y)
    z = y[labelled_set]
    index = np.where(z == 0)
    index = labelled_set[index]
    
    print(index, index.shape)
    X = X[index]
    features, top_features = wf.most_common_word_features(X, 30)
    #print(features)
    #print(sorted(features.items(), key=operator.itemgetter(1)))
    print()
    print(top_features)
    sys.exit()
    
    
    cv=CountVectorizer(max_df=0.85)
    word_count_vector=cv.fit_transform(X)
    print("TOP wors from CV: ", list(cv.vocabulary_.keys())[:10])
    
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    # you only needs to do this once, this is a mapping of index to 
    feature_names=cv.get_feature_names()
     
    # get the document that we want to extract keywords from
    #doc="When is the assignment due? q_mark"
    class_1_text = X[np.where(y == 1)].tolist()
    #x_all =X[unlabelled_set].tolist()
    print("Number of items in list : ",  len(class_1_text))
    #sys.exit()
     
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform(class_1_text))
     
    #sort the tf-idf vectors by descending order of scores
    sorted_items = wf.sort_coo(tf_idf_vector.tocoo())
     
    #extract only the top n; n here is 10
    keywords = wf.extract_topn_from_vector(feature_names,sorted_items,50)
     
    # now print the results
    #print("\n=====Doc=====")
    #print(doc)
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])

    
    