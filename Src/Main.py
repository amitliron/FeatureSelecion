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

def preprocessing(df, target_column_name):
    handle_empty_values(df)
    label_encoder(df)
    remove_text_from_input(df)
    handle_variance(df)
    feature_selection(df, target_column_name)

def run_model(df, target_column_name):

    from sklearn.model_selection import train_test_split
    X = df[df.columns[:-1]]
    y = df[target_column_name]
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


def plot_selection_result(lst, df):

    import matplotlib.pyplot as plt
    plt.close('all')

    X = df[df.columns[:-1]]
    all_features_names = X.columns.values

    dict = {}
    full_dict = {}

    for list_methods in lst:
        for res in list_methods:
            method_type = res[0]
            method_name = res[1]
            features_names = res[2]
            selected_features_names = [(el.strip()) for el in features_names]
            for selected_feature in selected_features_names:
                if selected_feature in dict:
                    count = dict[selected_feature]
                    count = count + 1
                    dict[selected_feature] = count

                    lst_methods_names = full_dict[selected_feature]
                    if method_name not in lst_methods_names:
                        lst_methods_names.append(method_name)
                    full_dict[selected_feature] = lst_methods_names
                else:
                    dict[selected_feature] = 1
                    full_dict[selected_feature] = [method_name]

    for feature in all_features_names:
        if feature not in dict:
            dict[feature] = 1

    plot_df = pd.DataFrame(dict, index=[0])
    plot_df.rename(index={0: "COUNT"}, inplace=True)
    df1_transposed = plot_df.T

    #print(df1_transposed)
    #df1_transposed.plot(kind='hist')
    #plt.xlabel("num of chosses")
    #plt.ylabel("features")
    #plt.show()

    # df1_transposed['COUNT'].plot(kind='bar')
    # plt.grid()
    # plt.show()

    #------------------------------------------------

    #print(full_dict)

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    plot_df_methods_names = pd.DataFrame(mlb.fit_transform(full_dict.values()), index=full_dict.keys(), columns=mlb.classes_)


    # plot_df_methods_names = pd.DataFrame()
    # row = 0
    # for feature in full_dict:
    #     for method in full_dict[feature]:
    #         if method not in plot_df_methods_names:
    #             plot_df_methods_names.insert(loc=0, column=method, value=[1])
    #     row = row + 1

    # for feature in full_dict:
    #
    #     if feature not in plot_df_methods_names.values:
    #         plot_df_methods_names.append(index=feature)

        # for method in full_dict[feature]:
        #     if method not in plot_df_methods_names:
        #         plot_df_methods_names.insert(loc=0, column=method, value=[1])
        # row = row + 1




    print("\n\n\n")
    print(plot_df_methods_names.index)
    print(plot_df_methods_names)
    #print("res: \n", plot_df_methods_names)
    #plot_df_methods_names.plot.scatter()
    #plt.scatter(plot_df_methods_names.preTestScore, df.postTestScore , s=df.age)
    #plot_df_methods_names.plot().scatter(x=plot_df_methods_names.values, y = plot_df_methods_names.columns)
    #plot_df_methods_names.reset_index().plot(kind='scatter', x='index', y=plot_df_methods_names.columns[1])
    #plt.show()
    print(plot_df_methods_names.shape)
    #plot_df_methods_names.plot(kind='bar')

    plt.scatter(*np.where(plot_df_methods_names)[::-1])
    plt.xticks(range(plot_df_methods_names.shape[1]), plot_df_methods_names.columns)
    plt.yticks(range(plot_df_methods_names.shape[0]), plot_df_methods_names.index)
    plt.show()


    #print(plot_df_methods_names.values)

    None

def print_freature_selection_result(lst):

    print("\nFeature Selection Result:")
    for list_methods in lst:
        for res in list_methods:
            method_type = res[0]
            method_name = res[1]
            features_names = res[2]
            features_names = [(el.strip()) for el in features_names]
            print(method_type, ",", method_name,",", features_names)

def feature_selection(df, target_column_name):

    X = df[df.columns[:-1]]
    y = df[target_column_name]

    result = []

    from FeatureSelection import filter_methods
    res = filter_methods.filter_methods(df, X, y, target_column_name)
    result.append(res)

    from FeatureSelection import wrapper_methods
    #result.append(wrapper_methods.wrapper_methods(df, X, y))

    from FeatureSelection import embedded_methods
    result.append(embedded_methods.embedded_methods(df, X, y))

    from FeatureSelection import hybrid_methods
    #res = hybrid_methods.hybrid_methods()
    #result.append(res)

    from FeatureSelection import genetic_algorithms_methods
    #res = genetic_algorithms_methods.genetic_algorithms_methods(df, X, y)
    # result.append(res)

    print_freature_selection_result(result)
    plot_selection_result(result, df)

def load_input():
    import os
    #file = os.getcwd() + '/Dataset/heart.csv'
    file = os.getcwd() + '/Dataset/nba_logreg.csv'

    df = pd.read_csv(file)
    #return df, "target"
    return df, "TARGET_5Yrs"

def main():
    df, target_column_name = load_input()
    preprocessing(df, target_column_name)
    #run_model(df, target_column_name)


if __name__ == "__main__":
    main()