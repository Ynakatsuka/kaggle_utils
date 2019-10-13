import math
from category_encoders.cat_boost import CatBoostEncoder
import numpy as np
import pandas as pd
from sklearn.base import clone
from .base import BaseFeatureTransformer


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


class OOFCategoryEncoder(BaseFeatureTransformer):
    def __init__(self, n_splits, cvfold, categorical_features, encoder=None, name='catboost_encoded'):
        self.n_splits = n_splits
        self.cvfold = cvfold
        self.categorical_features = categorical_features
        self.columns = [name + '_' + c for c in categorical_features]
        if encoder is None:
            self.encoder = CatBoostEncoder(
                cols=categorical_features,
                return_df=False,
            )
        else:
            self.encoder = encoder

    def fit_transform(self, X, y):
        self.cbe_ = []
        
        X_transformed = np.zeros((len(X), len(self.columns)), dtype=np.float32)
        
        for fold_id in range(self.n_splits):        
            train_index = self.cvfold['train_id' + str(fold_id)]==fold_id
            valid_index = self.cvfold.valid_id==fold_id

            self.cbe_.append(
                clone(self.encoder).fit(X.loc[train_index, self.categorical_features], y[train_index])
            )
            X_transformed[valid_index] = self.cbe_[-1].transform(
                X.loc[valid_index, self.categorical_features]
            )

        return pd.DataFrame(X_transformed, columns=self.columns)

    def transform(self, X):
        X_transformed = np.zeros((len(X), len(self.columns)), dtype=np.float32)
        for cbe in self.cbe_:
            X_transformed += cbe.transform(X) / self.n_splits
        return pd.DataFrame(X_transformed, columns=self.columns)

    
class ZValueOneHotEncoder(BaseFeatureTransformer):
    '''
    ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN
    "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"
    '''
    def __init__(self, filter_value=0.005, zvalue=5, m=0.5, categorical_features=[]):
        self.filter_value = filter_value
        self.zvalue = zvalue
        self.m = m
        self.categorical_features = categorical_features
        self.maxvalue_dict = {}
        
    def nan_check(self, x):
        if isinstance(x, float):
            if math.isnan(x):
                return True
        return False

    def encode_FE(self, dataframe, col):
        d = dataframe[col].value_counts(dropna=False)
        dataframe[col + '_FE'] = dataframe[col].map(d) / d.max()
        return dataframe

    def encode_OHE(self, dataframe, y, col):
        cv = dataframe[col].value_counts(dropna=False)
        cvd = cv.to_dict()
        vals = len(cv)
        th = self.filter_value * len(dataframe)
        sd = self.zvalue * 0.5/ math.sqrt(th)
        n = []; ct = 0; d = {}
        for x in cv.index:
            try:
                if cv[x]<th: 
                    break
                sd = self.zvalue * 0.5 / math.sqrt(cv[x])
            except:
                if cvd[x]<th: 
                    break
                sd = self.zvalue * 0.5 / math.sqrt(cvd[x])
            if self.nan_check(x): 
                r = y[dataframe[col].isna()].mean()
            else: 
                r = y[dataframe[col]==x].mean()
            if abs(r - self.m) > sd:
                nm = col + '_BE_' + str(x)
                if self.nan_check(x): 
                    dataframe[nm] = (dataframe[col].isna()).astype(np.uint8)
                else: 
                    dataframe[nm] = (dataframe[col]==x).astype(np.uint8)
                n.append(nm)
                d[x] = 1
            ct += 1
            if (ct+1)>=vals: 
                break

        return dataframe

    def fit_transform(self, dataframe, y):
        if self.categorical_features is None:
            self.categorical_features = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        for c in self.categorical_features:
            dataframe = self.encode_OHE(dataframe, y, c)
        return dataframe
    
    def transform(self, dataframe):
        raise NotImplementedError
    
    def get_categorical_features(self):
        return self.categorical_features
