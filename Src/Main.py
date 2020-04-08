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




def remove_text_from_input(df):
    for column in df:
        if df[column].dtype == object:
            df.drop([column],axis=1, inplace=True)


def label_encoder(df):
    print("[ERROR] label_encoder TBD")
    None

def feature_scaling(df):
    y = df.iloc[:,-1]
    target_name = df.columns[-1]
    from FeatureScaling import feature_scaling
    df = feature_scaling.scale(df.iloc[:,0:-1], standardize=False)
    df[target_name] = y
    return df

def feature_selection(df):
    from FeatureSelection import feature_selection
    feature_selection.feature_selection(df)


def preprocessing(df):
    handle_empty_values(df)
    label_encoder(df)
    remove_text_from_input(df)
    #return test_python(df)
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


    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')

    if config['Dataset']['random'] == "True":
        from Dataset import CreateRandomDataset as cd
        df = cd.generate_dataset()
        return df

    if config['Dataset']['iris'] == "True":
        from sklearn import datasets
        samples = datasets.load_iris()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        return df

    if config['Dataset']['breast_cancer'] == "True":
        from sklearn import datasets
        samples = datasets.load_breast_cancer()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        return df

    if config['Dataset']['nba'] == "True":
        file = 'winequality-white.csv'

    if config['Dataset']['wine'] == "True":
        from sklearn import datasets
        samples = datasets.load_wine()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        return df
        #file = 'winequality-white.csv'

    if config['Dataset']['sonar'] == "True":
        file = 'sonar.all-data.csv'

    import os
    full_location = os.getcwd() + '/../Dataset/' + file
    df = pd.read_csv(full_location, sep=";")
    return df


def test_python(df):
    from SklearnTests import SklearnTests
    #SklearnTests.test_score_validate_function(df)
    #SklearnTests.test_pipeline(df)
    SklearnTests.test_pca(df)
    SklearnTests.test_custome_filter(df)

def main():
    df = load_input()
    preprocessing(df)
    #run_model(df, target_column_name)


if __name__ == "__main__":
    main()