import functools
import gc
import json
import os
import time
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(ABC, BaseEstimator, TransformerMixin):
    '''
    data_dir: ../data/
    save, load: ../data/working/feature/
    config: ../data/working/config/
    '''
    def __init__(self, data_dir='../data/', name='', param_dict=None):
        self.data_dir = data_dir
        self.param_dict = param_dict
        self.name = self.__class__.__name__ + name
        self.feature_path = Path(data_dir) / 'working' / 'features' / self.name + '.ftr'
        self.config_path = Path(data_dir) / 'working' / 'config' / self.name + '.json'

    @staticmethod
    def check_path(path):
        if os.path.exists(path):
            print('Features of this version already exists.')
            return False
        else:
            return True

    @abstractmethod
    def transform(self, dataframe):
        raise NotImplementedError

    @abstractmethod
    def get_categorical_features(self):
        raise NotImplementedError

    @abstractmethod
    def get_numerical_features(self):
        raise NotImplementedError

    @abstractmethod
    def get_other_features(self):
        raise NotImplementedError

    @staticmethod
    def save_config(param_dict, path):
        with open(path, 'w') as f:
            json.dump(param_dict, f)

    @staticmethod
    def save(dataframe, path):
        dataframe.to_feather(path)

    def transform_decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.check_path(self.feature_path):
                res = func(self, *args, **kwargs)
                print('Saving...')
                self.save_config(self.param_dict, self.config_path)
                self.save(res, self.feature_path)
            else:
                print('Loading...')
                with open(self.config_path) as f:
                    old_param_dict = json.load(f)
                if old_param_dict != self.param_dict:
                    raise ValueError('new params were different from old one.')
                res = pd.read_feather(self.feature_path)
            return res
        return wrapper
    

class CategoricalEncoder(BaseFeatureTransformer):
    def __init__(self, data_dir='../data/', name='', param_dict=None, categorical_columns=[]):
        super().__init__(data_dir, name, param_dict)
        self.categorical_columns = categorical_columns

    def fit(self, X):
        return self

    def transform(self, dataframe):
        for c in self.categorical_columns:
            dataframe[c], uniques = pd.factorize(dataframe[c])
        return dataframe
    
    def get_categorical_features(self):
        return self.categorical_columns

    def get_numerical_features(self):
        return []

    def get_other_features(self):
        return []
