from itertools import combinations

import numpy as np
import pandas as ps
from sklearn.metrics import normalized_mutual_info_score

from .base import *
from .category_embedding import *
from .category_encoding import *
from .graph import *
from .groupby import *
from .image import *
from .image_pretrained import *
from .row_aggregations import *
from .text import *


def merge_columns(dataframe, columns):
    new_column = '_'.join(columns)
    dataframe[new_column] = ''
    for c in columns:
        dataframe[new_column] += dataframe[c].astype(str).fillna('null')
        
    return dataframe


def merge_columns_with_mutual_info_score(dataframe, columns, threshold=0.3):
    for c1, c2 in combinations(columns, 2):
        if normalized_mutual_info_score(dataframe[c1], dataframe[c2], average_method='arithmetic') > threshold:
            dataframe = merge_columns(dataframe, [c1, c2])
    return dataframe


def get_interactions(dataframe, interaction_features):
    for (c1, c2) in combinations(interaction_features, 2):
        dataframe[c1 + '_mul_' + c2] = dataframe[c1] * dataframe[c2]
        dataframe[c1 + '_div_' + c2] = dataframe[c1] / dataframe[c2]
    return dataframe


def get_time_features(dataframe, time_column):
    dataframe[time_column] = pd.to_datetime(dataframe[time_column])
    dataframe[time_column+'_day'] = dataframe[time_column].dt.day
    dataframe[time_column+'_dayofweek'] = dataframe[time_column].dt.dayofweek
    dataframe[time_column+'_weekofyear'] = dataframe[time_column].dt.weekofyear
    dataframe[time_column+'_is_weekend'] = (dataframe[time_column].dt.weekday>=5).astype(np.uint8)
    dataframe[time_column+'_month'] = dataframe[time_column].dt.month
    dataframe[time_column+'_hour'] = dataframe[time_column].dt.hour
    dataframe[time_column+'_minute'] = dataframe[time_column].dt.minute
    dataframe[time_column+'_second'] = dataframe[time_column].dt.second
    return dataframe
