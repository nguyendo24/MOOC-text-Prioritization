# -*- coding: utf-8 -*-
#
# author: amanul haque
#

from sklearn.feature_selection import SelectKBest, chi2

class feature_selection:
    
    def select_k_best(self, X_train, y_train, X_test):
        ch2 = SelectKBest(chi2, k= 'all')
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        return X_train, X_test
        