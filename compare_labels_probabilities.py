# -*- coding: utf-8 -*-
#
# author: amanul haque
#
class compare_labels_probabilities:
    
    def __init__(self):
        
        self.x= 0 
        
    def compare(self, pred_probs, pred_labels):
        index = []
        i = 0
        for label_prob, label in zip(pred_probs, pred_labels):
            label_prob = label_prob.tolist()
            if(label != label_prob.index(max(label_prob[0], label_prob[1]))):
                print("Mismatch, Predicted label Label : ", label, " probabilities :", label_prob)
                index.append(i)
                i+=1
            '''
            else:
                print(label_prob, "\t", (max(label_prob[0], label_prob[1])), "\t", label)
            '''
        return index, len(index)