# -*- coding: utf-8 -*-
#
## author: amanul haque
#

from nltk.corpus import stopwords
import pandas as pd
import string
import nltk
import sys
import numpy as np

class data_preprocessing_3:
    
    def __init__(self):
        self.output_file_path = ""
        self.input_file_path = ""
    
    def write_to_text_file(self, text):
        f = open("processed_text.txt", "w")
        text = " ".join(text)
        f.write(text)
        f.close()
    
    def read_from_text_file(self):
        f = open("processed_text.txt", "r")
        comments = f.read()
        f.close()
        return nltk.word_tokenize(comments)
    
    def remove_stopwords(self, tokenized_words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        #print(stopwords.words('english'))
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    def remove_punctuations(self, tokenized_words):
        new_words = []
        punctuation = set(string.punctuation)
        punctuation.remove("?")
        for word in tokenized_words:
            if word not in punctuation:
                new_words.append(word)
            
        return new_words        
                
    def to_lower_case(self, tokenized_words):
        new_words = []
        for word in tokenized_words:
            if word.isalpha():
                new_words.append(word.lower())
            else:
                new_words.append(word)
        return new_words
    
    def remove_digits(self, tokenized_words):
        new_words = []
        for word in tokenized_words:
            if word.isdigit():
                new_words.append("DIGIT")
            else:
                new_words.append(word)
        return new_words
    
    def replace_question_marks(self, tokenized_words):
        new_words = []
        for word in tokenized_words:
            if word == '?':
                new_words.append('q_mark')
            else:
                new_words.append(word)
        return new_words
        
    def preprocess_text(self, X):
        new_X = []
        for cmt in X:
            words = nltk.word_tokenize(str(cmt))
            comments = self.to_lower_case(words)
            #print("Initial word count : ", len(comments))
            comments = self.remove_stopwords(comments)
            #print("After removing stop words ", len(comments))
            
            comments = self.remove_punctuations(comments)
            #print("After removing punctautions : ", len(comments))
            
            comments = self.remove_digits(comments)
            #print("After removing digits ", len(comments))
            comments = self.replace_question_marks(comments)
            
            comments = " ".join(comments)
            new_X.append(comments)
            
        return np.array(new_X)
            


if __name__ == '__main__':
    
    tp = data_preprocessing_3()
    
    data = pd.read_csv("final_data/output_experiment.csv")
    comments = []
    i = 0
    for cmt in data['Content']:
        i+=1
        words = nltk.word_tokenize(str(cmt))
        comments.extend(words)
    
    print("Number of records : ", i)
    
    comments = tp.to_lower_case(comments)
    print("Initial word count : ", len(comments))
    comments = tp.remove_stopwords(comments)
    print("After removing stop words ", len(comments))
    
    comments = tp.remove_punctuations(comments)
    print("After removing punctautions : ", len(comments))
    
    comments = tp.remove_digits(comments)
    print("After removing digits ", len(comments))
    
    #tp.write_to_text_file(comments)
    #comment = read_from_text_file()
    
    fdist1 = nltk.FreqDist(comments)
    print(fdist1.most_common(50))