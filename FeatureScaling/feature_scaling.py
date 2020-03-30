from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale(df, nromalize=True, standardize=True):

    if nromalize==True:
        scaler = MinMaxScaler()
        res = scaler.fit_transform(df)

    if standardize==True:
        scaler = StandardScaler()
        scaler.fit(df)
        res = scaler.transform(df)

    return pd.DataFrame(res, columns=df.columns)