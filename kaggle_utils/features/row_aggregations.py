import gc
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d, _num_samples

    
class RowAggregationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, n_jobs=-1, pre_dispatch='2*n_jobs', **kwargs):
        super().__init__()
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        
    def _aggregate_row(self, row):
        non_zero_values = row[row.nonzero()]
        if len(non_zero_values)==0:
            aggregations = {'non_zero_mean': np.nan,
                            'non_zero_std': np.nan,
                            'non_zero_max': np.nan,
                            'non_zero_min': np.nan,
                            'non_zero_sum': np.nan,
                            'non_zero_skewness': np.nan,
                            'non_zero_kurtosis': np.nan,
                            'non_zero_median': np.nan,
                            'non_zero_q1': np.nan,
                            'non_zero_q3': np.nan,
                            'non_zero_log_mean': np.nan,
                            'non_zero_log_std': np.nan,
                            'non_zero_log_max': np.nan,
                            'non_zero_log_min': np.nan,
                            'non_zero_log_sum': np.nan,
                            'non_zero_log_skewness': np.nan,
                            'non_zero_log_kurtosis': np.nan,
                            'non_zero_log_median': np.nan,
                            'non_zero_log_q1': np.nan,
                            'non_zero_log_q3': np.nan,
                            'non_zero_count': np.nan,
                            'non_zero_fraction': np.nan
                            }
        else:
            aggregations = {'non_zero_mean': non_zero_values.mean(),
                            'non_zero_std': non_zero_values.std(),
                            'non_zero_max': non_zero_values.max(),
                            'non_zero_min': non_zero_values.min(),
                            'non_zero_sum': non_zero_values.sum(),
                            'non_zero_skewness': skew(non_zero_values),
                            'non_zero_kurtosis': kurtosis(non_zero_values),
                            'non_zero_median': np.median(non_zero_values),
                            'non_zero_q1': np.percentile(non_zero_values, q=25),
                            'non_zero_q3': np.percentile(non_zero_values, q=75),
                            'non_zero_log_mean': np.log1p(non_zero_values).mean(),
                            'non_zero_log_std': np.log1p(non_zero_values).std(),
                            'non_zero_log_max': np.log1p(non_zero_values).max(),
                            'non_zero_log_min': np.log1p(non_zero_values).min(),
                            'non_zero_log_sum': np.log1p(non_zero_values).sum(),
                            'non_zero_log_skewness': skew(np.log1p(non_zero_values)),
                            'non_zero_log_kurtosis': kurtosis(np.log1p(non_zero_values)),
                            'non_zero_log_median': np.median(np.log1p(non_zero_values)),
                            'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
                            'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75),
                            'non_zero_count': len(non_zero_values),
                            'non_zero_fraction': len(non_zero_values) / len(row)
                            }
    
        return np.array([*aggregations.values()])
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        parallel = Parallel(
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose
        )
        stats_list = parallel(delayed(self._aggregate_row)(X[i, :]) for i in range(len(X)))
        return np.array(stats_list)
