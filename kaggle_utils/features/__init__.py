from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
from .base import *
from .category_embedding import *
from .category_encoding import *
from .graph import *
from .groupby import *
from .image import *
# from .image_pretrained import *
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
