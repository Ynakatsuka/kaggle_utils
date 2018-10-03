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


class CategoricalEncoder(BaseFeatureTransformer):
    def __init__(self, categorical_feature=[]):
        self.categorical_feature = categorical_feature
        self.maxvalue_dict = {}

    def transform(self, dataframe):
        if self.categorical_feature is None:
            self.categorical_feature = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        for c in self.categorical_feature:
            dataframe[c], uniques = pd.factorize(dataframe[c])
            self.maxvalue_dict[c] = dataframe[c].max() + 1
        return dataframe
    
    def get_categorical_features(self):
        return self.categorical_feature
