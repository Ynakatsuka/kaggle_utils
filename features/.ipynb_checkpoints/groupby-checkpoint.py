import gc
import numpy as np
import pandas as pd
from .base import BaseFeatureTransformer
from ..utils import change_dtype


class BaseGroupByTransformer(BaseFeatureTransformer):
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict['key']
        if 'var' in p_dict.keys():
            var = p_dict['var']
        else:
            var = self.var
        if 'agg' in p_dict.keys():
            agg = p_dict['agg']
        else:
            agg = self.agg
        return key, var, agg

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=key)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()
        

class GroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['channel'], 
                'agg': ['count', 'nunique', 'cumcount']
            }
        ]
    '''

    
class LagGroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['time'], 
            }
        ]
    '''

    def __init__(self, param_dict=None, shift=1, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.shift = shift
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['lag']

    def _aggregate(self, dataframe):
        self.features = []
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[var].shift(-self.shift) - dataframe[var]
            features.columns = new_features
            features = features.fillna(self.fill_na)
            for c in new_features:
                if features[c].dtype == 'float64':
                    features[c] = features[c].astype('float32')
                else:
                    features[c] = features[c].dt.seconds.astype('float32')
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=False)

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.shift), v, 'groupby'] + key) for v in var for a in agg]


class EWMGroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['time'], 
            }
        ]
    '''

    def __init__(self, param_dict=None, alpha=0.5, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.alpha = alpha
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['ewm']

    def calc_shifted_ewm(self, series, adjust=True):
        return series.shift().ewm(alpha=self.alpha, adjust=adjust).mean()

    def _aggregate(self, dataframe):
        self.features = []
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[var].transform(self.calc_shifted_ewm)
            features = features.fillna(self.fill_na)
            features.columns = new_features
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=False)

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.alpha), v, 'groupby'] + key) for v in var for a in agg]


class BayesianMeanGroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['time'], 
            }
        ]
    '''

    def __init__(self, param_dict=None, l=10):
        super().__init__(param_dict)
        self.l = l
        self.agg = ['bayesian_mean']
        for p in param_dict:
            if len(p['var']) > 1:
                raise ValueError('len(var) must be 1.')

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict) 
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(['count', 'mean']).reset_index()
            features.columns = key + ['count', 'mean']
            features = dataframe[all_features].merge(features, how='left', on=key)
            #    len(new_features) == 1
            features[new_features[0]] = ((features[var[0]] / features['mean']) * 
                features['count'] + self.l) / (features['count'] + self.l)
            features = change_dtype(features, columns=new_features)
            self.features.append(features[new_features])
        return self

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=False)

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.l), v, 'groupby'] + key) for v in var for a in agg]


class TargetEncodingTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
            }
        ]
    '''

    def __init__(self, target, n_splits, cvfold, len_train, param_dict=None):
        super().__init__(param_dict)
        self.n_splits = n_splits
        self.cvfold = cvfold
        self.len_train = len_train
        self.var = [target]
        self.agg = ['encoding']

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key)
            features = dataframe[all_features].groupby(key)[
                var].agg('mean').reset_index()
            features.columns = key + new_features
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        return self

    def transform(self, dataframe):
        train = dataframe[:self.len_train]
        test = dataframe[self.len_train:]
        new_features = self.get_feature_names()
        result = pd.DataFrame(
            np.empty([len(dataframe), len(self.param_dict)]), columns=new_features)
        
        # for valid data
        for fold_id in range(self.n_splits):
            train_index = np.array(cvfold != fold_id)
            valid_index = np.array(cvfold == fold_id)
            trn = train.loc[train_index, key + [self.var]]
            val = train.loc[valid_index, key]

            local_avg = trn[self.var].mean()
            self._aggregate(trn)
            val = self._merge(val, merge=True)
            result.iloc[:len_train].loc[valid_index] = val[new_features].fillna(local_avg)

        # for test data
        global_avg = train[self.var].mean()
        self._aggregate(train)
        test = self._merge(test, merge=True)
        result.iloc[len_train:] = test[new_features].fillna(global_avg)

        return pd.concat([dataframe, result], axis=1)


class BayesianTargetEncodingTransformer(TargetEncodingTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
            }
        ]
    '''  

    def __init__(self, target, n_splits, cvfold, len_train, l=100, param_dict=None):
        super().__init__(target, n_splits, cvfold, len_train, param_dict)
        self.l = l
        # overwrite
        self.agg = ['bayesian_encoding']

    def _aggregate(self, dataframe):
        self.features = []
        local_avg = dataframe[self.var].mean()
        for param_dict in self.param_dict:
            key, var, agg = self._get_params(param_dict)
            all_features = list(set(key + self.var))
            new_features = self._get_feature_names(key)
            features = dataframe[all_features].groupby(key)[self.var].agg(
                ['sum', 'count']).reset_index()
            features.columns = key + ['sum', 'count']
            features[new_features] = (features['sum'] + self.l * local_avg) / (features['count'] + self.l)
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        return self

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.l), v, 'groupby'] + key) for v in var for a in agg]
