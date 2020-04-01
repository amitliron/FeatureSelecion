from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from FeatureSelection import filter_methods

class CustomeHybridFilter(TransformerMixin, BaseEstimator):

    def __init__(self):
        """
        constructor
        """
        super().__init__()
        self.clf = SVC()

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        self.clf.fit(X,y)
        pass

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        self.clf.transform(X, y)
        pass

    # def fit_transform(self, X, y=None, **kwargs):
    #     """
    #     perform fit and transform over the data
    #     :param X: features - Dataframe
    #     :param y: target vector - Series
    #     :param kwargs: free parameters - dictionary
    #     :return: X: the transformed data - Dataframe
    #     """
    #     self = self.fit(X, y)
    #     return 0#self.clftransform(X, y)

    def score(self, X, y):
        return 77