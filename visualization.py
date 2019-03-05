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

vectorizor = vectorization()
X_train, y_train, X_test, labelled_set, unlabelled_set, model = vectorizor.vectorize_text()
Y = pd.read_excel("sample_data.xlsx")['label']
print(X_train.shape)
print(y_train.shape)
X = np.concatenate((X_train, X_test),axis=0)
print(X.shape)

def plot_graph(X, Y):
    
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
    plt.scatter(center[:,0], center[:,1], marker = 'x' , color = 'r')
    
    for xy, cluster_l, label in zip(transformed_features, l, Y):
        col = ['red','blue','green']
        #print( xy, cluster_l, label)
        
        plt.scatter(xy[0],xy[1], c=col[label], alpha=0.9, s=30)
        plt.annotate(cluster_l,
                     xy=(xy[0], xy[1]),
                     xytext=(5, 3),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        
    #plt.savefig('plot.png', format='png', dpi=150, bbox_inches='tight')
    plt.show()
    
print(np.array(Y))   
plot_graph(X,Y)