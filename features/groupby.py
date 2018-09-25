import gc
from .basic import BaseFeatureTransformer
from ..utils import timer


class GroupbyTransformer(BaseFeatureTransformer):
    '''
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['channel'], 
                'agg':['count', 'nunique', 'cumcount']
            }
        ]
    '''
    def __init__(self, data_dir='../data/', name='', param_dict=None):
        super().__init__(data_dir, name, param_dict)
        self.features = []
        
    def fit(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            all_features = list(set(key + var))
            new_features = ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]
            features = dataframe[all_features].groupby(key)[var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self
        
    @timer
    @BaseFeatureTransformer.transform_decorator
    def transform(self, dataframe):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            dataframe = dataframe.merge(features, how='left', on=key)
        return dataframe[self.get_feature_names()].astype('float32')

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names
        
    def get_categorical_features(self):
        return []

    def get_numerical_features(self):
        return self.get_feature_names()

    def get_other_features(self):
        return []

    
class GroupbyDiffTransformer(BaseFeatureTransformer):
    '''
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['channel'], 
                'agg':['count', 'nunique', 'cumcount']
            }
        ]
    '''
    def __init__(self, data_dir='../data/', name='', param_dict=None, use_diffs_only=True):
        super().__init__(data_dir, name, param_dict)
        self.use_diffs_only = use_diffs_only
        self.features = []
        
    def fit(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            all_features = list(set(key + var))
            new_features = ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]
            features = dataframe[all_features].groupby(key)[var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self
        
    @timer
    @BaseFeatureTransformer.transform_decorator
    def transform(self, dataframe):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            dataframe = dataframe.merge(features, how='left', on=key)
        return dataframe[self.get_feature_names()].astype('float32')

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]
    
    def _get_diff_feature_names(self, key, var, agg):
        return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg = param_dict['key'], param_dict['var'], param_dict['agg']
            if not self.use_diffs_only:
                self.feature_names += self._get_feature_names(key, var, agg)
            self.feature_names += self._get_diff_feature_names(key, var, agg)
        return self.feature_names
        
    def get_categorical_features(self):
        return []

    def get_numerical_features(self):
        return self.get_feature_names()

    def get_other_features(self):
        return []
    

class GroupbyPrevDiffTransformer(BaseFeatureTransformer):
    '''
        param_dict = [
            {'key': ['user'], 'var': 'click_time'},
        ]
    '''
    @timer
    @BaseFeatureTransformer.transform_decorator
    def transform(self, dataframe):
        old_columns = dataframe.columns
        for param_dict in self.param_dict:
            key, var = param_dict['key'], param_dict['var']
            all_features = list(set(key + [var]))
            new_feature = '_'.join(['prev_diff', var, 'groupby'] + key)
            dataframe[new_feature] = dataframe[all_features].groupby(key)[var].shift(-1) - dataframe[var]
        self.new_features = list(set(dataframe.columns) - set(old_columns))

        return dataframe[self.new_features].astype('float32')

    def get_categorical_features(self):
        return []

    def get_numerical_features(self):
        return self.new_features

    def get_other_features(self):
        return []


class GroupbyNextDiffTransformer(BaseFeatureTransformer):
    '''
        param_dict = [
            {'key': ['user'], 'var': 'click_time'},
        ]
    '''
    @timer
    @BaseFeatureTransformer.transform_decorator
    def transform(self, dataframe):
        old_columns = dataframe.columns
        for param_dict in self.param_dict:
            key, var = param_dict['key'], param_dict['var']
            all_features = list(set(key + [var]))
            new_feature = '_'.join(['next_diff', var, 'groupby'] + key)
            dataframe[new_feature] = dataframe[var] - dataframe[all_features].groupby(key)[var].shift(+1)

        self.new_features = list(set(dataframe.columns) - set(old_columns))

        return dataframe[self.new_features].astype('float32')

    def get_categorical_features(self):
        return []

    def get_numerical_features(self):
        return self.new_features

    def get_other_features(self):
        return []


class AddColumnsTransformer(BaseFeatureTransformer):
    '''
        add_columns_dict = [
            {'key': ['ip','hour'], 'var': 'user'},
        ]
    '''
    D = 2**26

    @timer
    @BaseFeatureTransformer.transform_decorator
    def transform(self, dataframe):
        old_columns = dataframe.columns
        for param_dict in self.param_dict:
            key, var = param_dict['key'], param_dict['var']
            dataframe[var] = ''
            for col in key:
                dataframe[var] += (dataframe[col].astype(str) + '_')
            dataframe[var] = dataframe[var].apply(hash) % D
        self.new_features = list(set(dataframe.columns) - set(old_columns ))

        return dataframe[self.new_features]

    def get_categorical_features(self):
        return self.new_features

    def get_numerical_features(self):
        return []

    def get_other_features(self):
        return []
