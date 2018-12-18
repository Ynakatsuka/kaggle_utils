import numpy as np
from scipy.special import erfinv
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
# from .preprocess import 


class GaussRankScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=0.001):
        self.epsilon = epsilon
        self.lower = -1 + self.epsilon
        self.upper =  1 - self.epsilon
        self.range = self.upper - self.lower

    def transform(self, X, y=None):
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min()==0).all()
        assert (j.max()==len(j) - 1).all()
        
        j_range = len(j) - 1
        self.divider = j_range / self.range
        
        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)
        
        return transformed
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

        
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