import gc
import numpy as np
import pandas as pd
from .base import BaseFeatureTransformer
from ..utils import change_dtype


class BaseGroupByTransformer(BaseFeatureTransformer):
    def __init__(self, param_dict=None):
        self.param_dict = param_dict
        self.features = []
        self.fitted = False
        
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
        if 'on' in p_dict.keys():
            on = p_dict['on']
        else:
            on = key
        return key, var, agg, on
    
    def _aggregate(self, dataframe):
        raise NotImplementedError
    
    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=on)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe
    
    def fit(self, dataframe):
        self._aggregate(dataframe)
        self.fitted = True
        
    def transform(self, dataframe):
        if not self.fitted:
            self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)
    
    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]
    
    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
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
    def _aggregate(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        return self

    
class DiffGroupbyTransformer(BaseGroupByTransformer):      
    def __init__(self, param_dict=None, additional_stats=None):
        super().__init__(param_dict)
        self.additional_stats = additional_stats
        
    def _aggregate(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            
            new_features, base_features = [], []
            for a in agg:
                for v in var:
                    if not isinstance(a, str):
                        new_feature = '_'.join(['diff', a.__name__, v, 'groupby'] + key)
                        base_feature = '_'.join([a.__name__, v, 'groupby'] + key)
                    else:
                        new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                        base_feature = '_'.join([a, v, 'groupby'] + key) 
                    new_features.append(new_feature)
                    base_features.append(base_feature)
            
            g = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            g.columns = key + base_features
            features = dataframe[all_features].merge(g, on=key, how='left')
            
            for base_feature, new_feature in zip(base_features, new_features):
                features[new_feature] = features[base_feature] - features[v]

            features = features[key+new_features]
            
            if self.additional_stats:
                additional_new_features = self._get_feature_names(key, new_features, agg, prefix=False)
                features = features.groupby(key)[
                    new_features].agg(self.additional_stats).reset_index()
                features.columns = key + additional_new_features
                
            self.features.append(features)
        return self
    
    def transform(self, dataframe):
        if len(self.features):
            dataframe = self._merge(dataframe, merge=True)
        else:
            for param_dict in self.param_dict:
                key, var, agg, on = self._get_params(param_dict)
                for a in agg:
                    for v in var:
                        if not isinstance(a, str):
                            new_feature = '_'.join(['diff', a.__name__, v, 'groupby'] + key)
                            base_feature = '_'.join([a.__name__, v, 'groupby'] + key)
                        else:
                            new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                            base_feature = '_'.join([a, v, 'groupby'] + key)    
                        dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg, prefix=True):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        if prefix:
            return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in _agg]
        else:
            return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(BaseGroupByTransformer):      
    def __init__(self, param_dict=None, additional_stats=None):
        super().__init__(param_dict)
        self.additional_stats = additional_stats
        
    def _aggregate(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            
            new_features, base_features = [], []
            for a in agg:
                for v in var:
                    if not isinstance(a, str):
                        new_feature = '_'.join(['ratio', a.__name__, v, 'groupby'] + key)
                        base_feature = '_'.join([a.__name__, v, 'groupby'] + key)
                    else:
                        new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                        base_feature = '_'.join([a, v, 'groupby'] + key) 
                    new_features.append(new_feature)
                    base_features.append(base_feature)
            
            g = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            g.columns = key + base_features
            features = dataframe[all_features].merge(g, on=key, how='left')
            
            for base_feature, new_feature in zip(base_features, new_features):
                features[new_feature] = features[base_feature] / features[v]

            features = features[key+new_features]
            
            if self.additional_stats:
                additional_new_features = self._get_feature_names(key, new_features, agg, prefix=False)
                features = features.groupby(key)[
                    new_features].agg(self.additional_stats).reset_index()
                features.columns = key + additional_new_features
                
            self.features.append(features)
        return self
    
    def transform(self, dataframe):
        if len(self.features):
            dataframe = self._merge(dataframe, merge=True)
        else:
            for param_dict in self.param_dict:
                key, var, agg, on = self._get_params(param_dict)
                for a in agg:
                    for v in var:
                        if not isinstance(a, str):
                            new_feature = '_'.join(['ratio', a.__name__, v, 'groupby'] + key)
                            base_feature = '_'.join([a.__name__, v, 'groupby'] + key)
                        else:
                            new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                            base_feature = '_'.join([a, v, 'groupby'] + key)    
                        dataframe[new_feature] = dataframe[base_feature] / dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg, prefix=True):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        if prefix:
            return ['_'.join(['ratio', a, v, 'groupby'] + key) for v in var for a in _agg]
        else:
            return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]
        

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
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[var].shift(-self.shift) - dataframe[var]
            features.columns = new_features
            for c in new_features:
                if features[c].dtype in ['float16', 'float32', 'float64']:
                    features[c] = features[c].astype('float32')
                else:
                    features[c] = features[c].dt.seconds.astype('float32')
            features = features.fillna(self.fill_na)
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self
    
    def transform(self, dataframe):
        if not self.fitted:
            self._aggregate(dataframe)
        return self._merge(dataframe, merge=False)

    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.shift), v, 'groupby'] + key) for v in var for a in agg]


class CategoryLagGroupbyTransformer(LagGroupbyTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
                'var': ['device'], 
            }
        ]
    '''
    def __init__(self, param_dict=None, shift=1, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.shift = shift
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['catlag']

    def _aggregate(self, dataframe):
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = (dataframe[all_features].groupby(key)[var].shift(-self.shift) - dataframe[var])
            features.columns = new_features
            for c in new_features:
                nan_index = features[c].isnull()
                features[c] = features[c].fillna(0).astype(bool).astype(np.float32)
                features[c][nan_index] = np.nan
            features = features.fillna(self.fill_na)
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self

    
class CategoryShareGroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['day'], 
                'var': ['device'], 
            }
        ]
    '''
    def __init__(self, param_dict=None, shift=1, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.shift = shift
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['category_share']

    def _aggregate(self, dataframe):
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            
            frac = dataframe[all_features].groupby(key+var).size().reset_index()
            frac.columns = key + var + ['frac']
            
            denom = dataframe[all_features].groupby(key).size().reset_index()
            denom.columns = key + ['denom']
            
            features = frac.merge(denom, on=key, how='inner')
            features[new_features[0]] = features['frac'] / features['denom']
            features = features[key + new_features]
            
            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self
    

class PrevCategoryShareGroupbyTransformer(CategoryShareGroupbyTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['day'], 
                'var': ['device'], 
                'on': ['prev_day'],
            }
        ]
    '''
    def __init__(self, param_dict=None, shift=1, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.shift = shift
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['category_share_diff']

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', left_on=on, right_on=key)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    
class CategoryShareRankGroupbyTransformer(BaseGroupByTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['day'], 
                'var': ['device'], 
            }
        ]
    '''
    def __init__(self, param_dict=None, shift=1, fill_na=-1, sort_features=None):
        super().__init__(param_dict)
        self.shift = shift
        self.fill_na = fill_na
        self.sort_features = sort_features
        self.agg = ['category_share_rank']

    def _aggregate(self, dataframe):
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            
            frac = dataframe[all_features].groupby(key+var).size().reset_index()
            frac.columns = key + var + ['frac']
            
            denom = dataframe[all_features].groupby(key).size().reset_index()
            denom.columns = key + ['denom']
            
            features = frac.merge(denom, on=key, how='inner')
            features[new_features[0]] = features['frac'] / features['denom']
            features = features[key + new_features]
            features[new_features[0]] = features.groupby(key)[new_features[0]].rank()

            features = change_dtype(features, columns=new_features)
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self
    

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
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[var].transform(self.calc_shifted_ewm)
            features = features.fillna(self.fill_na)
            features.columns = new_features
            self.features.append(features)
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return self
    
    def transform(self, dataframe):
        if not self.fitted:
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
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict) 
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
        if not self.fitted:
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
        '''
            Params
            ------
            target : list
        '''
        super().__init__(param_dict)
        if not isinstance(target, list):
            self.var = [target]
        else:
            self.var = target
        self.agg = ['encoding']
        self.n_splits = n_splits
        self.cvfold = cvfold
        self.len_train = len_train
        
    def _encode(self, dataframe, merge_dataframe, key, var, new_features, avg):
        g = dataframe[key + var].groupby(key)[var].agg('mean').reset_index()
        g.columns = key + new_features
        g = change_dtype(g, columns=new_features)
        g[new_features] = g[new_features].fillna(avg)
        merge_dataframe = merge_dataframe.merge(g, on=key, how='left')
        return merge_dataframe

    def _aggregate(self, dataframe):
        train = dataframe[:self.len_train]
        test = dataframe[self.len_train:]
        
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            feature = pd.DataFrame(np.empty([len(dataframe), len(self.var)]), columns=new_features)

            # for valid data
            for fold_id in range(self.n_splits):                
                train_index = np.array(self.cvfold['train_id' + str(fold_id)]==fold_id).flatten()
                valid_index = np.array(self.cvfold.valid_id==fold_id).flatten()

                trn = train.loc[train_index, key + var]
                val = train.loc[valid_index, key]
                local_avg = trn[self.var].mean()
                val = self._encode(trn, val, key, var, new_features, local_avg)
                feature.iloc[:self.len_train, :].loc[valid_index] = val[new_features].values

            # for test data
            global_avg = train[var].mean()
            test = self._encode(train, test, key, var, new_features, global_avg)
            feature.iloc[self.len_train:, :] = test[new_features].values
            self.features.append(feature)    
        return self
        
    def transform(self, dataframe):
        if not self.fitted:
            self._aggregate(dataframe)
        return self._merge(dataframe, merge=False)
    

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

    def _encode(self, dataframe, merge_dataframe, key, var, new_features, avg):
        g = dataframe[key + var].groupby(key)[self.var].agg(['sum', 'count']).reset_index()
        g.columns = key + ['sum', 'count']
        g[new_features[0]] = ((g['sum'] + (self.l * avg).values) / (g['count'] + self.l)).fillna(avg)
        g = change_dtype(g, columns=new_features)
        merge_dataframe = merge_dataframe.merge(g[key + new_features], on=key, how='left')
        return merge_dataframe
            
    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, str(self.l), v, 'groupby'] + key) for v in var for a in agg]
    
    
class Seq2DecTargetEncodingTransformer(TargetEncodingTransformer):
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
        super().__init__(target, n_splits, cvfold, len_train, param_dict)
        # overwrite
        self.agg = ['seq2dec_target_encoding']

    def _encode(self, dataframe, merge_dataframe, key, var, new_features, avg):
        g = dataframe[key + var].groupby(key)[self.var].apply(lambda x: ''.join(x.values.flatten().astype(str))).reset_index()
        g.columns = key + new_features
        g[new_features[0]] = g[new_features[0]].apply(lambda x: x[:1] + '.' + x[1:]).astype(float)
        merge_dataframe = merge_dataframe.merge(g, on=key, how='left')
        return merge_dataframe
    
    def _aggregate(self, dataframe):
        train = dataframe[:self.len_train].reset_index(drop=True)
        test = dataframe[self.len_train:].reset_index(drop=True)
        
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            feature = pd.DataFrame(np.empty([len(dataframe), len(self.var)]), columns=new_features)
            feature[new_features[0]] = np.nan

            # for valid data
            for fold_id in range(self.n_splits):                
                train_index = np.array(self.cvfold['train_id' + str(fold_id)]==fold_id).flatten()
                valid_index = np.array(self.cvfold.valid_id==fold_id).flatten()

                trn = train.loc[train_index, key + var].reset_index(drop=True)
                val = train.loc[valid_index, key].reset_index(drop=True)
                val = self._encode(trn, val, key, var, new_features, None)
                
                feature.iloc[:self.len_train, :].loc[valid_index] = val[new_features].values

            # for test data
            test = self._encode(train, test, key, var, new_features, None)
            feature.iloc[self.len_train:, :] = test[new_features].values
            self.features.append(feature)    
        return self
            
    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]

    
class EWMTargetEncodingTransformer(TargetEncodingTransformer):
    '''
        Example
        -------
        param_dict = [
            {
                'key': ['ip','hour'], 
            }
        ]
    '''  
    def __init__(self, target, n_splits, cvfold, len_train, param_dict=None, alpha=0.5, fill_na=-1, sort_features=None):
        super().__init__(target, n_splits, cvfold, len_train, param_dict)
        # overwrite
        self.agg = ['ewm_target_encoding']
        self.alpha = alpha
        self.fill_na = fill_na
        self.sort_features = sort_features
        
    def calc_shifted_ewm(self, series, adjust=True):
        return series.shift().ewm(alpha=self.alpha, adjust=adjust).mean()

    def _encode(self, dataframe, merge_dataframe, key, var, new_features, avg):
        groupby = dataframe[key + var].groupby(key)
        g = groupby[self.var].apply(self.calc_shifted_ewm)
        keys = pd.DataFrame(groupby.groups.keys(), columns=key)
        g = pd.concat([keys, g], axis=1)
        g.columns = key + new_features
        merge_dataframe = merge_dataframe.merge(g, on=key, how='left')
        return merge_dataframe
    
    def _aggregate(self, dataframe):
        train = dataframe[:self.len_train].reset_index(drop=True)
        test = dataframe[self.len_train:].reset_index(drop=True)
        
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            feature = pd.DataFrame(np.empty([len(dataframe), len(self.var)]), columns=new_features)
            feature[new_features[0]] = np.nan

            # for valid data
            for fold_id in range(self.n_splits):                
                train_index = np.array(self.cvfold['train_id' + str(fold_id)]==fold_id).flatten()
                valid_index = np.array(self.cvfold.valid_id==fold_id).flatten()

                trn = train.loc[train_index, key + var].reset_index(drop=True)
                val = train.loc[valid_index, key].reset_index(drop=True)
                val = self._encode(trn, val, key, var, new_features, None)
                
                feature.iloc[:self.len_train, :].loc[valid_index] = val[new_features].values

            # for test data
            test = self._encode(train, test, key, var, new_features, None)
            feature.iloc[self.len_train:, :] = test[new_features].values
            self.features.append(feature)    
        return self
            
    def _get_feature_names(self, key, var, agg):
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in agg]
    