import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def func1():
    df = pd.DataFrame({'WEIGHT': [1, 3001, 3003, 2997],
                       'PRICE': [1, 3, 2, 5]},
                      index=['Orange', 'Apple', 'Banana', 'Grape'])

    print("befoer")
    print(df)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    res = scaler.fit_transform(df)
    # df[["A", "B"]] = min_max_scaler.fit_transform(df[["A", "B"]])
    print("after")
    print(res)


def func2():
    d = {'col1': [1, 2, 3, 4, 1, 2, np.nan, 4],
         'col2': [1, np.nan, 3, 4, np.nan, 2, 3, 4],
         'col3': [1, 550, 3, 4, np.nan, 2, 3, 4],
         'col4': [1, np.nan, 3, 4, 1, 2, 3, 5],
         'col5': [1, 2, 3, 4, np.nan, 2, 3, np.nan],
         'cls':  [1, 0, 1, 1, 1, 0, 0, 1]}

    df = pd.DataFrame(data=d)
    print("before")
    print(df.head(5))
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    z = np.abs(stats.zscore(df))
    print(np.where(z > 3))

    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    res = imputer.fit_transform(X)
    print("after")
    print(res)

    df2 = pd.DataFrame(data=res)
    z = np.abs(stats.zscore(df2))
    print(np.where(z > 3))

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(max_iter=10, random_state=0)

    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, y)

def func3():
    dict = {'COL1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 15],
            # 'COL2': [np.nan, 1150, 1160, 1117, 1131],
            # 'COL3': [np.nan, 20, 3035, 2, 50],
            'COL4': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15]}

    df = pd.DataFrame(dict)
    print(df.head())
    zscore = np.abs(stats.zscore(df))
    print(zscore)
    #df = df[(np.abs(stats.zscore(df, nan_policy='omit')) < 1.8).all(axis=1)]
    df = df[(np.nan_to_num(np.abs(stats.zscore(df, nan_policy='omit')), 0) <1.8).all(axis=1)]
    print(df)
    None


def main():
    func3()


if __name__ == "__main__":
    print("main")
    main()