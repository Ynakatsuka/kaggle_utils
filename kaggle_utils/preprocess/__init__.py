import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin


def to_category(dataframe, cat=None):
    if cat is None:
        cat = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    for c in cat:
        dataframe[c], uniques = pd.factorize(dataframe[c])
    return dataframe

        
class BoxCoxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
        
    def inverse_boxcox(y, ld):
        if ld == 0:
            return(np.exp(y))
        else:
            return(np.exp(np.log(ld*y+1)/ld))
    
    def transform(self, X, y=None):
        self.fitted = True
        X, self.la = boxcox(X)
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
        
    def inverse_transform(self, X):
        if self.fitted:
            return self.inverse_boxcox(X, self.la)
        else:
            raise NotFittedError()
