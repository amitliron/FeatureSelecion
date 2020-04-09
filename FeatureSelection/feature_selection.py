import pandas as pd
import numpy as np


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
        if feature not in full_dict:
            full_dict[feature] = []


    dfPlot = pd.DataFrame()
    for list_methods in lst:
        for res in list_methods:
            method_name = res[1]
            features_names = res[2]
            selected_features_names = [(el.strip()) for el in features_names]
            for selected_feature in selected_features_names:
                dfPlot.at[method_name,selected_feature] =  1


    '''
        dict:       key   =  feature
                    value =  count
                    
        full_dict   key   = feature
                    value = list(algs)    
                    
                    
        dfPlot:
                  columns: features
                  rows:    methods               
    
    '''

    '''
        FIRST FIGURE 
        Histogram
        features -> counter
    '''
    plot_df = pd.DataFrame(dict, index=[0])
    plot_df.rename(index={0: "COUNT"}, inplace=True)
    df1_transposed = plot_df.T
    #print(df1_transposed)
    #df1_transposed.plot(kind='hist')
    plt.figure(1)
    plt.title("Features to Target (Number Of Chooses)")
    plt.xlabel("num of chosses")
    plt.ylabel("features")
    df1_transposed['COUNT'].plot(kind='bar')
    plt.grid()
    #plt.show()

    '''
        SECOND FIGURE
        Features - > Alg Selected
        
        X: Features
        Y: Methods
    '''

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    plot_df_methods_names = pd.DataFrame(mlb.fit_transform(full_dict.values()), index=full_dict.keys(), columns=mlb.classes_)
    plt.figure(2)
    plt.title("Corleation Between Features")
    plt.scatter(*np.where(plot_df_methods_names)[::-1])
    plt.xticks(range(plot_df_methods_names.shape[1]), plot_df_methods_names.columns)
    plt.yticks(range(plot_df_methods_names.shape[0]), plot_df_methods_names.index)


    '''
        Third Figure
        
        X: Features
        Y: Methods (all of them)
    
    '''

    plt.figure(3)
    dfPlot.fillna(value=0, inplace=True)
    plt.scatter(*np.where(dfPlot)[::-1])
    plt.xticks(range(dfPlot.shape[1]), dfPlot.columns)
    plt.yticks(range(dfPlot.shape[0]), dfPlot.index)


    # show all plot
    plt.show()

def print_feature_selection_result(lst):

    print("\nFeature Selection Result:")
    for list_methods in lst:
        for res in list_methods:
            method_type = res[0]
            method_name = res[1]
            features_names = res[2]
            features_names = [(el.strip()) for el in features_names]
            print("\n",method_type, ",", method_name,":\n", features_names)



def feature_selection(df):

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    result = []

    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')

    if config['Feature Selection']['filter_method']=="True":
        from FeatureSelection import filter_methods
        result.append(filter_methods.filter_methods(df, X, y))

    if config['Feature Selection']['wrapper_method'] == "True":
        from FeatureSelection import wrapper_methods
        result.append(wrapper_methods.wrapper_methods(df, X, y))

    if config['Feature Selection']['embedded_method'] == "True":
        from FeatureSelection import embedded_methods
        result.append(embedded_methods.embedded_methods(df, X, y))

    if config['Feature Selection']['hybrid_method'] == "True":
        from FeatureSelection import hybrid_methods
        result.append(hybrid_methods.hybrid_methods(df, X, y))

    if config['Feature Selection']['genetic_method'] == "True":
        from FeatureSelection import genetic_algorithms_methods
        result.append(genetic_algorithms_methods.genetic_algorithms_methods(df, X, y))

    print_feature_selection_result(result)
    plot_selection_result(result, df)