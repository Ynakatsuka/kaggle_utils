from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(ABC, BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def __call__(self, dataframe):
        return self.transform(dataframe)

    def fit(self, X):
        return self 

    @abstractmethod
    def transform(self, dataframe):
        raise NotImplementedError

    def get_categorical_features(self):
        return [] 

    def get_numerical_features(self):
        return []
