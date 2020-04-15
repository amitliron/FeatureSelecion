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
    df = feature_scaling.scale(df.iloc[:,0:-1])
    df[target_name] = y
    return df

def feature_selection(df):
    from FeatureSelection import feature_selection
    feature_selection.feature_selection(df)

def preprocessing(df):

    print_statistics(df)
    handle_empty_values(df)
    label_encoder(df)
    remove_text_from_input(df)
    df = feature_scaling(df)
    feature_selection(df)


def print_statistics(df):
    X = df[df.columns[:-1]]
    X = X.values
    for col in range(X.shape[1]):
        mean = X[:, col].mean()
        var = X[:, col].var()
        ratio = var / mean
        print("Column: ", df.columns[col]," Mean: ", mean, " var = ", var, " var/mean: ", ratio)
    print("")


def add_deep_learning_prediction(classifier_list, scores_result, df):

    # check if we need to be here
    if 'deep_learning' not in classifier_list:
        return

    # add all imports
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD, Adam
    from sklearn.preprocessing import LabelEncoder
    from keras.layers import Dropout

    # prepare data
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]


    # hot encoding
    encoder = LabelEncoder()
    y1 = encoder.fit_transform(y)
    y = pd.get_dummies(y1).values

    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # normazlie
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # build model
    model = Sequential()
    model.add(Dense(81, activation='relu', input_shape = ((df.shape[1]-1),)))

    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(150, activation='relu', input_shape=((df.shape[1] - 1),)))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])

    # validation_split=0.3
    model.fit(X_train, y_train,epochs=120)
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    scores_result['deep_learning'] = score
    None


    # save results
    scores_result[type(model).__name__] = model.score(X_test, y_test)



def run_model(df):

    from sklearn.model_selection import train_test_split
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    classifier_list = get_classifiers_list()
    scores_result = {}

    add_deep_learning_prediction(classifier_list, scores_result, df)

    for classifier_name in classifier_list:
        classifier = create_classifier(classifier_name)
        classifier.fit(X, y)
        scores_result[type(classifier).__name__ ] = classifier.score(X_test, y_test)

    max_predict_value = max(scores_result.values())
    min_predict_value = min(scores_result.values())

    import matplotlib.pylab as plt
    fig, ax = plt.subplots()
    lists = sorted(scores_result.items())
    classifier_name, classifier_score = zip(*lists)
    barlist = plt.bar(classifier_name, classifier_score)
    if df.index.name is not None:
        plt.title("Dataset: " + df.index.name)
    for i, v in enumerate(classifier_score):
        txt = "{:.2f}".format(v)
        plt.text(i, v-0.05, txt, color='blue', va='center', fontweight='bold')
        if v == max_predict_value:
            barlist[i].set_color('g')
        elif v == min_predict_value:
            barlist[i].set_color('r')
    plt.show()


def get_classifiers_list():

    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')

    res = []
    if config['Classifier']['random_forest'] == "True":
        res.append('random_forest')

    if config['Classifier']['k_neighbors'] == "True":
        res.append('k_neighbors')

    if config['Classifier']['svm'] == "True":
        res.append('svm')

    if config['Classifier']['naive_bayes'] == "True":
        res.append('naive_bayes')

    if config['Classifier']['logistic_regression'] == "True":
        res.append('logistic_regression')

    if config['Classifier']['deep_learning'] == "True":
        res.append('deep_learning')

    return res


def create_classifier(classifier_name = None):

    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')

    classifier = None
    if classifier_name=='random_forest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()

    if classifier_name=='k_neighbors':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5)

    if classifier_name=='svm':
        from sklearn import svm
        classifier = svm.SVC()

    if classifier_name=='naive_bayes':
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()

    if classifier_name=='logistic_regression':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver='lbfgs')


    return classifier



def load_input():


    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')

    if config['Dataset']['random'] == "True":
        from Dataset import CreateRandomDataset as cd
        df = cd.generate_dataset()
        df.index.name = "Random"
        return df

    if config['Dataset']['iris'] == "True":
        from sklearn import datasets
        samples = datasets.load_iris()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        df.index.name = "Iris"
        return df

    if config['Dataset']['breast_cancer'] == "True":
        from sklearn import datasets
        samples = datasets.load_breast_cancer()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        df.index.name = "breast cancer"
        return df

    if config['Dataset']['wine'] == "True":
        from sklearn import datasets
        samples = datasets.load_wine()
        X = samples.data
        y = samples.target
        df = pd.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y
        df.index.name = "Wine"
        return df

    name = "None"
    if config['Dataset']['sonar'] == "True":
        file = 'sonar.all-data.csv'
        name = 'sonar'

    elif config['Dataset']['diabetes'] == "True":
        file = 'diabetes.csv'
        name = 'diabetes'
        sep = ","

    elif config['Dataset']['nba'] == "True":
        file = 'winequality-white.csv'
        name = 'nba'

    import os
    full_location = os.getcwd() + '/../Dataset/' + file
    df = pd.read_csv(full_location, sep=sep)
    df.index.name = name
    print("----------------------")
    print("shape: ", df.shape)
    print("dtypes: \n", df.dtypes)
    print("head: \n", df.head(3))
    print("----------------------")
    return df


def test_python(df):
    from SklearnTests import SklearnTests
    #SklearnTests.test_score_validate_function(df)
    #SklearnTests.test_pipeline(df)
    SklearnTests.test_pca(df)
    SklearnTests.test_custome_filter(df)

def main():
    df = load_input()
    #preprocessing(df)
    run_model(df)


if __name__ == "__main__":
    main()