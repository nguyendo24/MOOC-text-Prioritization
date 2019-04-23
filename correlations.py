# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code finds relationship between parameters.
#                   Both parameters need to be of same length in form of a list, numpy array or pandas series (or any other iterable 1-d structure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class correlations:
    
    def __init__(self):
        self.input_file_path = "Data/finla_survey_data_results.csv"
        
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
    x, y, z, = df['aggregate_Urgency'], df['aggregate_Complexity'], df['aggregate_Clarity']
    
    print("Pearson correlation: ", np.corrcoef(x,y))