import pandas as pd



'''
    embedded -> 2 methods:
        1. regularization
        2. tree base


    L1 regularization        - > reduce coefficients  to zero
    L2 regularization        - > coefficients approaching zero
    L1/L2 regularization     - > combination of the L1 and L2  
    
'''

def random_forest(df, X, y):

    x_train = X
    y_train = y

    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    model = SelectFromModel(RandomForestClassifier(n_estimators=100))

    # fit the model to start training.
    model.fit(x_train, y_train)
    selected_feat = X.columns[(model.get_support())]

    '''
    READ ME:
    1. when we usr from SelectFromModel we can use get_support() 
    2. otherwise we usr model.feature_importances_
    3. in order to get score we need to rerun with test samples and use score function (score (x_test, y_test))
    '''

    # get the importance of the resulting features.
    #importances = model.feature_importances_
    #print('embedded_methods (random_forest ) Optimal number of features: ', len(importances))
    #print('embedded_methods (random_forest ) Optimal number of features: ', importances)
    #print('embedded_methods (random_forest ) Feature importance: ', model.get_support())
    #print('embedded_methods (random_forest ) Feature importance: ', selected_feat)
    #pd.series(model.estimator_, feature_importances_,.ravel()).hist()
    return selected_feat

def lasso(df, X, y):

    x_train = X
    y_train = y

    # using logistic regression with penalty l1.
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    estimator = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    selection = SelectFromModel(estimator)
    selection.fit(x_train, y_train)
    #print(selection.estimator_.score(x_train, y_train))

    # see the selected features.
    selected_features = x_train.columns[(selection.get_support())]
    return selected_features



def embedded_methods(df, X, y):
    lst1 = lasso(df, X, y)
    lst2 = random_forest(df, X, y)

    lst = []
    lst.append(('embedded', 'lasso', lst1))
    lst.append(('embedded', 'RF', lst2))
    return lst