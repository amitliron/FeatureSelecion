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


def use_select_k_best(func, df, X, y):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import SelectPercentile
    selector = SelectKBest(func, k='all').fit(X, y)
    selector = SelectPercentile(func, percentile=90).fit(X, y)
    res = selector.transform(X)

    support = np.asarray(selector.get_support())
    columns_with_support = X.columns[support]

    return list(columns_with_support)

def remove_low_variance(df, X, y):

    # convert ti ndarray
    #X = X.values

    '''
        var[x] = p(1-p)
    '''

    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

    # for col in range(X.shape[1]):
    #     mean = X[:, col].mean()
    #     var = X[:, col].var()
    #     ratio = var / mean
    #     print("Mean: ", mean, " var = ", var, " rate: ", ratio)

    sel.fit(X)
    support = np.asarray(sel.get_support())
    columns_with_support = X.columns[support]

    return list(columns_with_support)


def filter_methods(df, X, y):
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import mutual_info_classif



    #kolmogorov_smirnov(df, X, y)
    res1 = remove_low_variance(df, X, y)
    res2 = use_select_k_best(chi2, df, X, y)
    res3 = use_select_k_best(f_classif, df, X, y)
    res4 = use_select_k_best(mutual_info_classif, df, X, y)
    res5 = correlation_between_features_to_them_self(df, max_threshold=0.4)


    res = []
    res.append(('filter', 'low_variance', res1))
    res.append(('filter', 'chi2', res2))
    res.append(('filter', 'ANOVA ', res3))
    res.append(('filter', 'Mutual information', res4))
    res.append(('filter', 'cross (redundant)', res5))
    return res
