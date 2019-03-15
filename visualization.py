# -*- coding: utf-8 -*-
#
# author: amanul haque
#

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from word_to_vec_vectorizer import vectorization
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
from main_file import main_file

class visualtization:
    
    def __init__(self):
        
        self.input_file_path = "output_file.csv"
        self.output_file_path = "output_file_2.csv"
    
    def plot_graph(self, X, Y):
        
        tsne_init = 'pca'  # could also be 'random'
        tsne_perplexity = 10.0
        tsne_early_exaggeration = 1.0
        tsne_learning_rate = 0.001
        random_state = 1
    
        model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
                 early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)
    
        transformed_features = model.fit_transform(X)
        true_k = 2
        km = KMeans(n_clusters = true_k)
        km.fit(transformed_features)
        l = km.labels_
        #print(l)
        dic = defaultdict(list)
        #print(dic)
        
        for x,y in zip(Y,l):
            dic[y].append(x)
    
        for key,value in dic.items():
            print('Cluster: ---------------- ')
            print(key)
            print('\n')
            print(value)
            print('\n')
    
        center = km.cluster_centers_
        
        #print(center[:,0], center[:,1])
        Y= Y.astype(int)
        plt.scatter(center[:,0], center[:,1], marker = 'x' , color = 'r')
        
        for xy, cluster_l, label in zip(transformed_features, l, Y):
            
            #print(type(xy), type(xy[0]), type(cluster_l), type(label))
            col = ['red','blue','green']
            #print( xy, cluster_l, label)
            
            plt.scatter(xy[0],xy[1], c=col[label], alpha=0.9, s=30)
            plt.annotate(cluster_l,
                         xy=(xy[0], xy[1]),
                         xytext=(5, 3),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
            
        plt.savefig('plot_word2vec.png', format='png', dpi=150, bbox_inches='tight')
        plt.show()
        
if __name__ == '__main__':
    
    v = visualtization()
    mf = main_file()
    
    df = pd.read_csv(mf.input_file_path, sep=',')
    X, y = mf.get_input_text_and_label(df)
    y, labelled_set, unlabelled_set = mf.get_test_train_split(X, y)
    
    print("X shape ", X.shape)
    #X_train, y_train, X_test = mf.get_vectorized_data(X, y, labelled_set, unlabelled_set)
    X_train, y_train, X_test = vectorization().word2vec_vectorization(X, y, labelled_set, unlabelled_set)
    print(X_train.shape, type(X_train))
    print(y_train.shape, type(y_train))
    X = np.concatenate((X_train, X_test),axis=0)
    #print(np.array(y_train))   
    v.plot_graph(X_train, y_train)