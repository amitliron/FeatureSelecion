from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale(df):


    import configparser
    config = configparser.ConfigParser()
    config.read('../Configuration/Configuration.ini')
    res = None
    if config['Feature Scaling']['nromalize'] == "True":
        scaler = MinMaxScaler()
        res = scaler.fit_transform(df)

    elif config['Feature Scaling']['standardize'] == "True":
        scaler = StandardScaler()
        scaler.fit(df)
        res = scaler.transform(df)

    if res is None:
        return df

    return pd.DataFrame(res, columns=df.columns)