# -*- coding: utf-8 -*-
#
# author: amanul haque
#

import pandas as pd
import numpy as np
import datetime
import sys

class metadata:
    
    def __init__(self):
        
        self.input_file_path = "output_file.csv"
    
    def get_deadline(self,index):
        #print("indexes are: ", index)
        data =  pd.read_csv(self.input_file_path).iloc[index]
        deadline = data['deadline'] 
        time_stamp = data['Time']
        
        date_diff_values = self.get_date_diff_values(time_stamp, deadline)
        return date_diff_values
        
        
    def get_date_diff_values(self, timestamps, deadlines):
        
        diff = []
        for d1, d2 in zip(timestamps, deadlines):
            
            #print("D1: ", d1, "\t", "D2 : ", d2)
            
            if(d2!='0'):
                
                date_time_obj = datetime.datetime.strptime(d1, '%m/%d/%Y %H:%M')
                d1 = date_time_obj.date()
                date_time_obj = datetime.datetime.strptime(d2, '%m/%d/%Y')
                d2 = date_time_obj.date()
                d = (d2-d1).days
                if(d <= 1):
                    diff.append(1)
                elif(d<=2):
                    diff.append(0.75)
                elif(d<=5):
                    diff.append(0.4)
                elif(d<7):
                    diff.append(0)
                else:
                    diff.append(-0.2)
                    
            else:
                diff.append(0)
        
        return np.array(diff)
            
    
    def calculate_deadline_weight(self, index):
        return self.get_deadline(index)
        