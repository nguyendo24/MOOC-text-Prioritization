# -*- coding: utf-8 -*-
#
# author: Amanul Haque
#
# File Description: This code does standard NLP data-preprocessing like: lemmatization, stop-word removal, stemming etc.


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import string
import nltk
import sys
import numpy as np
import math

class data_preprocessing_3:
    
    def __init__(self):
        self.output_file_path = ""
        self.input_file_path = ""
        #self.remove_list = ['constructor', 'name', 'something', 'figure', 'create', 'inhericdoc', 'complete', 'variable', 'boundary', 'issue', 'instantiate', 'miss', 'facultyrecordsio', 'write', 'sum', 'cast', 'test', 'multiple', 'anyone', 'must', 'problem', 'second', 'properly', 'correctly', 'false', 'weird', 'try', 'bugreader', 'caught', 'birthpredator', 'parameter', 'coverage', 'combination', 'occur', 'methods', 'ts_coursemanagertest', 'cheat', 'command', 'summary', 'is', 'birthprey', 'directly', 'test_movies', 'child', 'bugtrackermodel', 'need', 'todo', 'catch', 'error', 'confuse']
        self.remove_list = ['q_mark']
        
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
                new_words.append("digit")
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
    
    def stemming(self, tokenized_words):
        ps = PorterStemmer()
        new_words = []
        for word in tokenized_words:
            new_words.append(ps.stem(word))
        return new_words
    
    def lematization(self, tokenized_words):
        
        wordnet_lemmatizer = WordNetLemmatizer()        
        pos_tags = nltk.pos_tag(tokenized_words)
        new_words = []
        for tup in pos_tags:
            word, tag = tup[0], tup[1]
            if(word.isalpha()):
                word = word.lower()
            if(tag in ['VB', 'VBD', 'VBG','VBN','VBP','VBZ']):
                tag = wordnet.VERB
            elif (tag in ['JJ', 'JJR','JJS']):
                tag = wordnet.ADJ 
            elif (tag in ['RB', 'RBR', 'RBS']):
                tag = wordnet.ADV
            else:
                tag = wordnet.NOUN
                        
            #print(word, tag)
            new_words.append(wordnet_lemmatizer.lemmatize(word,pos=tag))
			
        return new_words
    
    def remove_words(self, tokenized_words, remove_list):
        new_words = []
        for word in tokenized_words:
            if word not in remove_list:
                new_words.append(word)
        return new_words
            
        
            
    def preprocess_text(self, X):
        new_X = []
        for cmt in X:
            words = nltk.word_tokenize(str(cmt))
            comments = self.to_lower_case(words)
            comments = self.stemming(comments)
            
            '''
            
            #print("Initial word count : ", len(comments))
            comments = self.remove_stopwords(comments)
            #print("After removing stop words ", len(comments))
            
            comments = self.remove_punctuations(comments)
            #print("After removing punctautions : ", len(comments))
            
            comments = self.remove_digits(comments)
            #print("After removing digits ", len(comments))
            comments = self.replace_question_marks(comments)
            
            #comments = self.stemming(comments)
            
            comments = self.lematization(comments)
            
            comments = self.remove_words(comments, self.remove_list)
            '''
            
            comments = " ".join(comments)
            if(len(comments) == 0):
                #print("NAN detected")
                new_X.append("empty_text")
            else:
                new_X.append(str(comments))    
            
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