import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
    SelectKBest - require K as input
'''

def kolmogorov_smirnov(df, X, y, min_threshold=0.3):
    from scipy.stats import kstest
    from scipy.stats import ks_2samp

    b = y
    for i in range(0, len(X.columns)):
        a = X.iloc[:, i]
        res = ks_2samp(a, b)
        print(res)
    None


def mutal_information_between_features_and_target(df, X, y, min_threshold=0.3, show_plot=False):
    from sklearn.feature_selection import mutual_info_classif
    mi_res = mutual_info_classif(X, y)

    lst = []
    for i in range(len(mi_res)):
        if mi_res[i] >= min_threshold:
            lst.append(df.columns[i])

    return lst


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
        if abs(corrlation[i]) >= min_threshold:
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


def use_select_k_best(func, df, X, y, min_threshold=0.3):

    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(func, k='all').fit(X, y)
    res = selector.transform(X)
    feature_names = list(X.columns.values)
    lst = []
    for i in range(len(selector.scores_)):
        if abs(selector.scores_[i]) >= min_threshold:
            lst.append(feature_names[i])

    print(selector.scores_)
    return lst


def filter_methods(df, X, y):
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import mutual_info_classif

    #kolmogorov_smirnov(df, X, y)
    res1 = mutal_information_between_features_and_target(df, X=X, y=y, min_threshold=0.2)
    res2 = correlation_between_features_and_target(df, min_threshold=0.6)
    res3 = correlation_between_features_to_them_self(df, max_threshold=0.4)
    res4 = use_select_k_best(chi2, df, X, y, min_threshold=0.6)
    res5 = use_select_k_best(f_classif, df, X, y, min_threshold=0.6)
    res6 = use_select_k_best(mutual_info_classif, df, X, y, min_threshold=0.6)

    res = []
    #res.append(('filter', 'MI\nbetween\nfeatures\nand\ntarget', res1))
    #res.append(('filter', 'correlation\nbetween\nfeatures\nand\ntarget', res2))
    #res.append(('filter', 'correlation\nbetween\nfeatures\nto_them\nself', res3))
    res.append(('filter', 'chi2', res4))
    res.append(('filter', 'ANOVA ', res5))
    res.append(('filter', 'Mutual information', res6))
    return res
