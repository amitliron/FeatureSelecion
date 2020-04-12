
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rfecv(df, X, y, plot_graph=True):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFECV

    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=3, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X, y)

    #features = np.where(rfecv.support_ == True)[0]
    features = X.columns[rfecv.support_ == True]
    #print('wrapper_methods (rfecv )Optimal number of features: {}'.format(rfecv.n_features_))
    #print('wrapper_methods (rfecv )Optimal number of features: ', )

    if plot_graph==True:
        dset = pd.DataFrame()
        dset['attr'] = X.columns
        dset['importance'] = pd.Serias(rfecv.estimator_.feature_importances_)

        dset = dset.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(16, 14))
        plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
        plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Importance', fontsize=14, labelpad=20)
        plt.show()

    return features


def forward_backward_selection(df, X, y, forward=True, floating=False):

    from mlxtend.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestClassifier

    sfs = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features="best",
                                    forward=forward,
                                    floating=floating,
                                    scoring='accuracy',
                                    cv=2,
                                    n_jobs=-1)

    sfs = sfs.fit(X, y)
    selected_features = X.columns[list(sfs.k_feature_idx_)]


    # if forward==True:
    #     print("wrapper_methods (Forward): ", selected_features)
    # else:
    #     print("wrapper_methods (Backward: ", selected_features)

    return selected_features



def wrapper_methods(df, X, y):
    list1 = forward_backward_selection(df, X, y, forward=True)
    list2 = forward_backward_selection(df, X, y, forward=False)
    list3 = forward_backward_selection(df, X, y, forward=True)
    list4 = forward_backward_selection(df, X, y, forward=True)
    list5 = rfecv(df, X, y, plot_graph=False)
    res = []
    res.append(('wrapper', 'forward selection', list1))
    res.append(('wrapper', 'backward selection', list2))
    res.append(('wrapper', 'step floating FS', list3))
    res.append(('wrapper', 'step floating BS', list4))
    res.append(('wrapper', 'RFECV', list5))

    # REFCV
    # Recursive Feature Elimination CV
    # 1. run model (RF)
    # 2. remove least importance feature and run again
    # 3. if score decrease -> save this feature

    # Step floating forward selection
    # 1. select best feature
    # 2. select worst
    # 3. check if score improve or decrease by deleting worst feature

    return res