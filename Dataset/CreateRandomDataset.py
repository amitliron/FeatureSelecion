
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import pandas as pd
import numpy as py


def generate_dataset(plot=False, num_of_features=40):

    n_redundant = int(num_of_features*0.15)
    n_informative = num_of_features - n_redundant
    # X - array of shape[n_samples, n_features]
    # y - array of shape [n_samples] (labels)
    # flip_y - noise
    X1, Y1 = make_classification(flip_y=0.35, weights = [0.65, 0.35], n_features=num_of_features, n_redundant=n_redundant, n_informative=n_informative,n_classes=2,n_clusters_per_class=3, n_samples=500)
    None

    if plot==True:
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
        plt.subplot(322)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
        plt.show()

    df = pd.DataFrame(data=X1)

    col_names = []
    for i in range(0, num_of_features):
        col_names.append('COL_' + str(i))
    df.columns = col_names
    df['Taregt'] = Y1
    return df, X1, Y1
