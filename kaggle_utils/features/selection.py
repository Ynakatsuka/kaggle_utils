import lightgbm as lgb
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from .base import BaseFeatureTransformer


class FeatureSelector(BaseFeatureTransformer):
    def __init__(
        self, feature_names, target, len_train,
        task_type=None, method='gain', n_repeats=5, model=None,
    ):
        self.feature_names = feature_names
        self.target = target
        self.len_train = len_train
        self.task_type = task_type
        self.method = method
        self.n_repeats = n_repeats
        self.model = model
        self.drop_features = []
    
    def fit(self, dataframe):
        X = dataframe[self.feature_names].iloc[:self.len_train]
        y = dataframe[self.target].iloc[:self.len_train]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
        if self.task_type is None:
            # binary, multiclass, multilabel-indicator, multiclass-multioutput, continuous
            self.task_type = type_of_target(y)
        
        # VarianceThreshold
        self.variance_transformer = VarianceThreshold()
        self.variance_transformer.fit(X)
        self.drop_features += np.array(self.feature_names)[
            self.variance_transformer.variances_==0
        ].tolist()
        
        if self.model is None:
            if self.task_type == 'binary' or self.task_type == 'multiclass':
                self.model = lgb.LGBMClassifier(objective=self.task_type, n_jobs=-1, num_iterations=10000)
            elif self.task_type == 'continuous':
                self.model = lgb.LGBMRegressor(objective='regression', n_jobs=-1, num_iterations=10000)
            else:
                raise ValueError('invalid task type.')
            self.model.fit(
                X_train, y_train, 
                early_stopping_rounds=100, 
                eval_set=[(X_valid, y_valid)],
                verbose=100,
            )
                
        if self.method == 'gain':
            if hasattr(self.model, 'feature_importance'):
                self.importances = self.model.feature_importance('gain')
            elif hasattr(self.model, 'feature_importances_'):
                self.importances = self.model.feature_importances_
            else:
                raise ValueError('invalid model type.')
            
            self.drop_features += np.array(self.feature_names)[
                self.importances==0
            ].tolist()
        elif self.method == 'permutation':
            self.importances = permutation_importance(
                self.model, X_train, y_train, n_repeats=self.n_repeats
            )['importances_mean']
            self.drop_features += np.array(self.feature_names)[
                self.importances==0
            ].tolist()
        
        return self
    
    def transform(self, dataframe):
        columns = list(set(dataframe.columns) - set(self.drop_features))
        return dataframe[columns]
    
    def fit_transform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)
