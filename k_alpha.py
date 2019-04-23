# -*- coding: utf-8 -*-
#
'''
Python implementation of Krippendorff's alpha -- inter-rater reliability

(c)2011-17 Thomas Grill (http://grrrr.org)

Python version >= 2.4 required
'''


from __future__ import print_function

try:
    import statistics
    import numpy as np    
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
    
    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.


if __name__ == '__main__': 
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data_ = pd.read_excel("Data/Survey_labels_only.xlsx")
    #Hana_u	Gabriel_u	Yiqiao_u	Michael_u Colton_u	Shailaja_u Effat_u Shreey_u Ammar_u Jannifer_u	Adam_u	Samiha_u


    #coder_1, coder_2, coder_3, coder_4 = list(data_['TA1_Urgency']), list(data_['TA2_Urgency']), list(data_['TA3_Urgency']), list(data_['TA4_Urgency'])    
    #coder_1, coder_2, coder_3, coder_4 = list(data_['TA1_Complexity']), list(data_['TA2_Complexity']), list(data_['TA3_Complexity']), list(data_['TA4_Complexity'])    
    coder_1, coder_2, coder_3, coder_4 = list(data_['Hana_u']), list(data_['Gabriel_u']), list(data_['Yiqiao_u']), list(data_['Michael_u'])
    coder_5, coder_6, coder_7, coder_8 = list(data_['Colton_u']), list(data_['Shailaja_u']), list(data_['Effat_u']), list(data_['Shreey_u'])
    coder_9, coder_10, coder_11, coder_12 = list(data_['Ammar_u']), list(data_['Jannifer_u']), list(data_['Adam_u']), list(data_['Samiha_u'])
    missing = '*' # indicator for missing values
    
    '''
    
    d = [coder_9, coder_10, coder_11, coder_12]
    for i in range(3):
        for j in range(i,4):
            c = [d[i], d[j]]
            print(i+1, j+1, "\t" , krippendorff_alpha(c, interval_metric, missing_items=missing))
            
        
    print("coder 1 : ", d[0])
    print("coder 2 : ", d[1])
    '''
    
    d1 = coder_1 + coder_5 + coder_9
    d2 = coder_2 + coder_6 + coder_10
    d3 = coder_3 + coder_8 + coder_11
    d = [d1, d2, d3]
    print(krippendorff_alpha(d, interval_metric, missing_items=missing))
    
    mean1, std1, mean2, std2, mean3, std3, mean4, std4 = np.mean(coder_1), np.std(coder_1), np.mean(coder_2), np.std(coder_2), np.mean(coder_3), np.std(coder_3), np.mean(coder_4), np.std(coder_4)
    mean5, std5, mean6, std6, mean7, std7, mean8, std8 = np.mean(coder_5), np.std(coder_5), np.mean(coder_6), np.std(coder_6), np.mean(coder_7), np.std(coder_7), np.mean(coder_8), np.std(coder_8)
    mean9, std9, mean10, std10, mean11, std11, mean12, std12 = np.mean(coder_9), np.std(coder_9), np.mean(coder_10), np.std(coder_10), np.mean(coder_11), np.std(coder_11), np.mean(coder_12), np.std(coder_12)
    
    print(mean1, std1, "\t", mean2, std2, "\t", mean3, std3, "\t", mean4, std4)
    print(mean5, std5, "\t", mean6, std6, "\t", mean7, std7, "\t", mean8, std8)
    print(mean9, std9, "\t", mean10, std10, "\t", mean11, std11, "\t", mean12, std12)
    
    
    
    data = (
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3", # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *", # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4", # coder C
    )
    

    array = [d.split() for d in data]  # convert to 2D list of string items
    #print(array)
    
    #print("nominal metric: %.3f" % krippendorff_alpha(d, nominal_metric, missing_items=missing))
    print("interval metric: %.3f" % krippendorff_alpha(d, interval_metric, missing_items=missing))