import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_empty_values(df):
    columns = list(df)
    for column in columns:
        if df[column].isna().sum() > 0:
            print(column + " contains: " + df[column].isna().sum() + " nulls")

def handle_empty_values(df):
    if  df.isnull().sum().sum() == 0:
        return

    for column in df:
        if df[column].isna().sum() > 0:
            rate = df[column].isna().sum() / df[column].count()
            if rate < 0.1:
                df[column].fillna(0, inplace=True)
            else:
                print("TBD")


def handle_variance (df):
    print("[ERROR] handle_variance - not implemented yet")
    #print(df.var())
    None

def remove_text_from_input(df):
    for column in df:
        if df[column].dtype == object:
            df.drop([column],axis=1, inplace=True)


def label_encoder(df):
    print("[ERROR] label_encoder TBD")
    None

def feature_scaling(df):
    from FeatureScaling import feature_scaling
    df = feature_scaling.scale(df, standardize=False)
    return df

def feature_selection(df):
    from FeatureSelection import feature_selection
    feature_selection.feature_selection(df)


def preprocessing(df):
    handle_empty_values(df)
    label_encoder(df)
    remove_text_from_input(df)
    return test_python(df)
    handle_variance(df)
    df = feature_scaling(df)
    feature_selection(df)

def run_model(df):

    from sklearn.model_selection import train_test_split
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    classifier = create_classifier('')
    classifier.fit(X, y)
    #predictions = model.predict(X_test)
    print("Model: " + type(classifier).__name__ + " Score: " + str(classifier.score(X_test, y_test)))
    None

def create_classifier(classifier_name):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)

    from sklearn import svm
    classifier = svm.SVC()

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()

    return classifier



def load_input():
    import os
    #file = os.getcwd() + '../Dataset/nba_logreg.csv'
    file = './../Dataset/nba_logreg.csv'
    df = pd.read_csv(file)
    return df


def test_python(df):
    from SklearnTests import SklearnTests
    #SklearnTests.test_score_validate_function(df)
    #SklearnTests.test_pipeline(df)
    SklearnTests.test_custome_filter(df)

def main():
    df = load_input()
    preprocessing(df)
    #run_model(df, target_column_name)


if __name__ == "__main__":
    main()