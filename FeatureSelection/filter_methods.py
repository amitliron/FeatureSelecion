import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
    SelectKBest - require K as input
'''

def correlation_between_features_and_target(df, min_threshold=0.3, show_plot=False):

    target_label = df.columns[-1]

    # print coraltion
    corrlation = df.drop(target_label, axis=1).apply(lambda x: x.corr(df[target_label]))
    #print(corrlation)
    feature_names = list(df.columns.values)

    # show graph
    if show_plot==True:
        del (feature_names[-1])
        indices = np.argsort(corrlation)
        plt.title('Features To Target Correlation')
        plt.barh(range(len(indices)), corrlation[indices], color='g', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Corrlation')
        plt.show()

    # return sublist
    lst = []
    for i in range(len(corrlation)):
        if corrlation[i] >= min_threshold:
            lst.append(feature_names[i])

    return lst

def correlation_between_features_to_them_self(df, max_threshold=0.3):
    feature_names = list(df.columns.values)
    del (feature_names[-1])

    dict = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
                corr = np.abs(df[df.columns[i]].corr(df[df.columns[j]]))
                if corr <= max_threshold and i not in dict:
                    dict.append(i)

    lst = []
    for i in range(len(list(dict))):
        lst.append(feature_names[i])
    #print("High Corrleations between features: ", lst)
    return lst


def filter_methods(df, X, y):
    res1 = correlation_between_features_and_target(df)
    res2 = correlation_between_features_to_them_self(df)
    res = []
    res.append(('filter', 'correlation\nbetween\nfeatures\nand\ntarget', res1))
    res.append(('filter', 'correlation\nbetween\nfeatures\nto_them\nself', res2))
    return res
