# -*- coding: utf-8 -*-
#
# author: amanul haque
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class correlations:
    
    def __init__(self):
        self.input_file_path = "Survey_results/binaried_correlation_data.csv"
        
    def get_data(self, filepath):
        data = pd.read_excel(filepath)
        return data['TA1_Urgency'], data['TA1_Complexity'], data['TA1_Clarity']
    		
    def plot_graph(self, x, y, z):
        plt.plot(range(0,len(x)), x)
        plt.plot(range(0,len(y)), y)
        plt.plot(range(0,len(z)), z)    
        plt.show()
        
    def binarize_variables(self, x):
        i = 0
        for el in x:
            if(el <= 2):
                x.iloc[i] = 0
            else:
                x.iloc[i] = 1
            i+=1
        return x
    
    def compare_individual_params(self, X, Y):
        result =[[0, 0],[0, 0]]
        for x, y in zip(X,Y):
            if(x == y and x == 0):      result[0][0]+=1
            elif(x == y and x == 1):    result[1][1]+=1
            elif(x !=y and x == 0):     result[0][1]+=1
            elif(x != y and x == 1):    result[1][0]+=1
        return np.array(result)
            
    
if __name__ == '__main__':
    
    c = correlations()
    df = pd.read_csv(c.input_file_path)
    x, y, z = df['Urgency'], df['Complexity'], df['Clarity']
    print(c.compare_individual_params(x,y))
    
    
    '''
    x, y, z = c.get_data(c.input_file_path)
    #c.plot_graph(x, y, z)
    x = c.binarize_variables(x)
    y = c.binarize_variables(y)
    z = c.binarize_variables(z)
    df = pd.DataFrame()
    df['Urgency'] = x
    df['Complexity'] = y
    df['Clarity'] = z
    print(df)
    print(type(x))
    #df.to_csv("Survey_results/binaried_correlation_data.csv")
    sys.exit()
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    plt.scatter(x,y)
    plt.show()
    print(np.corrcoef(x,y))
    print(np.corrcoef(x,z))
    print(np.corrcoef(z,y))
    #print(x, y, z)
    '''
