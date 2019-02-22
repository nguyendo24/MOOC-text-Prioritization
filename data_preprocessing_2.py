# -*- coding: utf-8 -*-
#
# author: amanul haque
#

import sys
import re
import csv
import pandas as pd
import numpy as np

class data_preprocessing_2:
    
    def __init__(self):
        
        self.input_file = 'final_data/processed_output.csv'
        self.output_file = 'final_data/output_experiment.csv'
    
    def detect_website_address(self, string):
        
        pattern = r"http[s]?:\/\/([-\w]+)([\.]+\w+)+(\/[-\w]+)*((\.\w+)?)\/?"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<WEBSITE_ADDRESS>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_website_address(string)
        
        return string
    
    def detect_assertion_errors(self, string):
        
        pattern = r"expected[\s:]?\s*<(.|[\r\n])*?>\s?but was[\s:]?\s*<(.|[\r\n])*?>"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<ASSERTION_ERROR>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_assertion_errors(string)
        
        return string
        
    def detect_comments(self, string):
        
        #Pattern to match both single and multiline comments in a java code
        pattern = r"\/\/(.*)|\/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+\/"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<COMMENT>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_comments(string)
        
        return string
    
    def detect_sop_statements(self, string):
        
        pattern = r"System.out.print(ln)?\(.*\)"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<SOP_STATEMENT>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_sop_statements(string)
        
        return string
     
    def detect_file_paths(self, string):
        
        #Pattern for file paths
        pattern = r"(\w+\/)(\w+\/)([-\w\/])+(\.\w+)?"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<FILE_PATH>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_file_paths(string)
        
        #Pattern for class path
        pattern = r"(\w+\.)(\w+\.)([-\w\.])+(\.\w+)?"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            print("matched string is : ", string[start_index:end_index], "\t", start_index, "\t", end_index)
            str_list = list(string)
            str_list = str_list[:start_index] + list('<FILE_PATH>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_file_paths(string)
        
        return string
      
    
    
    def detect_error_messages(self, string):
        
        pattern = r"(([-\w_\$]+)\.)+(\w)+(\([-\w_\$\.\s]+(:\d+)\))+"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            #print("matched string is : ", string[start_index:end_index])
            str_list = list(string)
            str_list = str_list[:start_index] + list('<ERROR_MESSAGE>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_error_messages(string)
        
        pattern = r"(\w+\.)+(\w+(Commands+|Error+|Exception+))"
        mt = re.search(pattern, string)
        if(mt != None):
            start_index = int(mt.start())
            end_index = int(mt.end())
            
            #print("matched string is : ", string[start_index:end_index])
            str_list = list(string)
            str_list = str_list[:start_index] + list('<JAVA_ERROR>') + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            
            return self.detect_error_messages(string)
        else:
            return string
        
    def identify_method_declerations(self, string, flag):
        
        pattern = r"\w+\("
        #print("Current string is : ", string)
        mt = re.search(pattern, string)
        str_list = list(string)
        if(mt != None and flag == 0):
            #print("found")
            #print("Match is ", mt)
            
            start_index = int(mt.start())
            end_index = int(mt.end())
            #print("start_index :", start_index)
            #print("end_index :", end_index)
            #print("matched string is  ", string[start_index:end_index])
            #end_index-=1
            #print(str_list[end_index])
            if(end_index >= len(str_list)):
                #str_list = str_list[:start_index] + list(' <METHOD_NAME> ') + str_list[end_index:]
                string = "".join(str_list[0:len(str_list)])
                return string
            elif(str_list[end_index-1] == '('):
                string, flag = self.code_parser(start_index, end_index, str_list, "<METHOD_NAME>", ['(',')'])
                #String ended without any closing brackets
                if(len(string) == 0):                
                    return string
                else:
                    #print("returned string is : ", string, " \t flag is ", flag )
                    string = self.identify_method_declerations(string, flag)
                    
                #print("String is: ", string)
        return(string)
    
    
    def identify_method_blocks(self, string, flag):
        
        pattern = r"<METHOD_NAME>\s+\{"
        #print("Current string is : ", string)
        mt = re.search(pattern, string)
        str_list = list(string)
        if(mt != None and flag == 0):
            #print("found")
            #print("Match is ", mt)
            
            start_index = int(mt.start())
            end_index = int(mt.end())
            #print("matched string is  ", string[start_index:end_index])
            #end_index-=1
            #print(str_list[end_index])
            if(end_index >= len(str_list)):
                #str_list = str_list[:start_index] + list(' <METHOD_NAME> ') + str_list[end_index:]
                string = "".join(str_list[0:len(str_list)])
                return string
            elif(str_list[end_index-1] == '{'):
                string, flag = self.code_parser(start_index, end_index, str_list, "<METHOD_BLOCK>", ['{','}'])
                #String ended without any closing brackets
                if(len(string) == 0):                
                    return string
                else:
                    #print("returned string is : ", string, " \t flag is ", flag )
                    string = self.identify_method_blocks(string, flag)
                    
                #print("String is: ", string)
        return(string)
       
    def identify_class_blocks_1(self, string, flag):
        
        pattern = r"(public|private)\s+(((\w+\s*)?)((\w+\s*)?)){"
        #print("Current string is : ", string)
        mt = re.search(pattern, string)
        str_list = list(string)
        if(mt != None and flag == 0):
            #print("found")
            #print("Match is ", mt)
            
            start_index = int(mt.start())
            end_index = int(mt.end())
            #print("matched string is  ", string[start_index:end_index])
            #end_index-=1
            #print(str_list[end_index])
            if(end_index >= len(str_list)):
                #str_list = str_list[:start_index] + list(' <METHOD_NAME> ') + str_list[end_index:]
                string = "".join(str_list[0:len(str_list)])
                return string
            elif(str_list[end_index-1] == '{'):
                string, flag = self.code_parser(start_index, end_index, str_list, "<CLASS_BLOCK>", ['{','}'])
                #String ended without any closing brackets
                if(len(string) == 0):                
                    return string
                else:
                    #print("returned string is : ", string, " \t flag is ", flag )
                    string = self.identify_class_blocks_1(string, flag)
                    
                #print("String is: ", string)
        return(string)
        
    def identify_class_blocks_2(self, string):
        
        pattern = r"(public|private)\s+((\w+\s*)?)((\w+\s*)?)((\w+\s*)?)<METHOD_BLOCK>"
        print("Current string is : ", string)
        if(re.search(pattern, string) != None):
            string = re.sub(pattern, r'<CODE_BLOCK>' ,string)
            return self.identify_class_blocks_2(string)
        return string
        
    def code_parser(self, start_index, end_index, str_list, annotation_name, bracket_style):
        
        stack = [bracket_style[0]]
        string = ""
        #print("end index :", end_index, "\t", str_list[end_index])
        #print("len : ", len(str_list), len(stack))
        while(end_index < len(str_list) and len(stack)>0):
            if(str_list[end_index] == bracket_style[0]):
                stack.append(bracket_style[0])
            elif(str_list[end_index] == bracket_style[1]):
               stack.pop()
            #print("Match is :", str_list[end_index])
            end_index+=1
        #sys.exit()
        
        if(len(stack) == 0 and end_index < len(str_list)):
            str_list = str_list[:start_index] + list(annotation_name) + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            flag = 0
        elif(len(stack) == 0 and end_index == len(str_list)):
            str_list = str_list[:start_index] + list(annotation_name) + str_list[end_index:]
            string = "".join(str_list[0:len(str_list)])
            flag = 1
        else:
            string = "".join(str_list[0:len(str_list)])
            flag = 2
        return string, flag
    
    
    def identify_code_blocks(self, string):
        
        string = self.identify_method_declerations(string, 0)
        string = self.identify_method_blocks(string, 0)
        string = self.identify_class_blocks_1(string, 0)
        string = self.identify_class_blocks_2(string)
        return string
            
    def process_data(self, X):
        
        processed_comments = []
        for cmt in X:
            #print(row['Content'])
            #print("index :", index)
        #if(index%2 == 0):
            #print("index :", index)
            print("Before Processing ", str(cmt))
            cmt_processed = self.detect_website_address(cmt)
            cmt_processed = self.detect_assertion_errors(cmt_processed)
            cmt_processed = self.detect_sop_statements(cmt_processed)
            cmt_processed = self.detect_comments(cmt_processed)
            cmt_processed = self.detect_error_messages(cmt_processed)
            cmt_processed = self.detect_file_paths(cmt_processed)
            cmt_processed = self.identify_code_blocks(cmt_processed)
            print("\n\nAfter processing : ", cmt_processed)
            processed_comments.append(cmt_processed)
            
        return np.array(processed_comments)

'''
#pattern for code line = [\s]?([-\.\(\)\w])*[\s]?=[\s]?.*[\s]?;       
ic = data_preprocessing_2()
data = pd.read_csv(ic.input_file)

# =============================================================================
# string = data['Content'][1387]
# #data = data.iloc[1387:1388,]
# print("Initial String : \n", string)
# print("Here")
# print("\n\nFinal String : \n", ic.detect_code_2(string,0))
# #string = "public void main(){ \n            }"
# 
# =============================================================================

df = ic.process_data(data)
df.to_csv(ic.output_file, sep=',')

'''
#print("final string is : \n'", string ,"'")
#print("match : ", mt.start(), mt.end(), mt.group())