import pandas as pd
import numpy as np


def test_pca(df):

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    from sklearn.decomposition import PCA
    #pca = PCA(n_components=2)
    pca = PCA(n_components  = 'mle', svd_solver = 'full')
    post_pca_array = pca.fit_transform(X)
    #print("PCA, components = ", pca.components_)
    print("PCA, shape = ", post_pca_array.shape)

    n_pcs = pca.components_.shape[0]
    print("PCA, components = ", n_pcs)

    initial_feature_names = df.columns[0:-1]
    print("Original Colums = ", initial_feature_names)
    most_important_index = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [initial_feature_names[most_important_index[i]] for i in range(n_pcs)]
    print("PCA, most_important_index = ", most_important_index)
    print("PCA, most_important_names = ", most_important_names)

    None

def test_custome_filter(df):

    from FeatureSelection import CustomeHybridFilter

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel

    steps = [('scaler', StandardScaler()),
             ('CustomeHybridFilter', CustomeHybridFilter.CustomeHybridFilter()),
             ]
    pipeline = Pipeline(steps)  # define the pipeline object.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    pipeline.fit(X_train, y_train)
    print("pipeline score: ", pipeline.score(X_test, y_test))


def test_score_validate_function(df):
    from sklearn import datasets, linear_model
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    lasso = linear_model.Lasso()

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf.fit(X_train, y_train)
    print("CV Score = ", clf.score(X_test, y_test))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train)
        print("K-FOLD Score = ", clf.score(X_test, y_test))

    print("cross_val_score = ", cross_val_score(clf, X, y, cv=2))
    print("cross_val_predict = ", cross_val_predict(clf, X, y, cv=2))


def test_pipeline(df):

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel

    steps = [('scaler', StandardScaler()),
             ('FeatureSelection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
             ('SVM', SVC())]
    pipeline = Pipeline(steps)  # define the pipeline object.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    parameteres = {'SVM__C': [0.001, 10e5],
                   'SVM__gamma':[0.1,0.01],
                   'FeatureSelection__estimator__C': [0.1, 1.0]}

    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=3, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)
    print("pipeline score: ", grid.score(X_test, y_test))
    print("pipeline best params: ", grid.best_params_)


    None


def test_VarianceThreshold():

    from sklearn import datasets
    from sklearn.feature_selection import VarianceThreshold
    import pandas as pds
    import numpy as np

    samples = datasets.load_iris()
    X = samples.data
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

    for col in range(X.shape[1]):
        mean = X[:,col].mean()
        var = X[:,col].var()
        ratio = var/mean
        print("Mean: ", mean, " var = ", var, " rate: ", ratio)

    sel.fit(X)
    print(sel.get_support())

    None

def main():
    test_VarianceThreshold()


if __name__ == "__main__":
    main()