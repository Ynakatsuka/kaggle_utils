import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseFeatureTransformer


class BasicTextFeatureTransformer(BaseFeatureTransformer):
    def __init__(self, text_columns):
        self.text_columns = text_columns
    
    def _get_features(self, dataframe, column):
        dataframe[column+'_num_chars'] = dataframe[column].apply(len)
        dataframe[column+'_num_capitals'] = dataframe[column].apply(lambda x: sum(1 for c in x if c.isupper()))
        dataframe[column+'_caps_vs_length'] = dataframe[column+'_num_chars'] / dataframe[column+'_num_capitals']
        dataframe[column+'_num_exclamation_marks'] = dataframe[column].apply(lambda x: x.count('!'))
        dataframe[column+'_num_question_marks'] = dataframe[column].apply(lambda x: x.count('?'))
        dataframe[column+'_num_punctuation'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        dataframe[column+'_num_symbols'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '*&$%'))
        dataframe[column+'_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        dataframe[column+'_num_unique_words'] = dataframe[column].apply(lambda x: len(set(w for w in x.split())))
        dataframe[column+'_words_vs_unique'] = dataframe[column+'_num_unique_words'] / dataframe[column+'_num_words']
        dataframe[column+'_num_smilies'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))
        return dataframe
    
    def transform(self, dataframe):
        dataframe[self.text_columns] = dataframe[self.text_columns].astype(str).fillna('missing')
        for c in self.text_columns:
            dataframe = self._get_features(dataframe, c)
        return dataframe


class TextVectorizer(BaseFeatureTransformer):
    def __init__(self, text_columns,
                 vectorizer=CountVectorizer(), 
                 transformer=TruncatedSVD(n_components=128),
                 name='count_svd'):
        self.text_columns = text_columns
        self.n_components = transformer.n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)
    
    def transform(self, dataframe):
        dataframe[self.text_columns] = dataframe[self.text_columns].astype(str).fillna('missing')
        features = []
        for c in self.text_columns:
            sentence = self.vectorizer.fit_transform(dataframe[c])
            feature = self.transformer.fit_transform(sentence)
            feature = pd.DataFrame(feature, columns=[self.name + f'_{i:03}' for i in range(self.n_components)])
            features.append(feature)
        dataframe = pd.concat([dataframe]+features, axis=1)
        return dataframe
    
    
class EmojiFeatureTransformer(BaseFeatureTransformer):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def transform(self, dataframe):
        dataframe[self.text_columns] = dataframe[self.text_columns].astype(str).fillna('missing')
        
        module_path = os.path.dirname(__file__)
        emoji1 = pd.read_csv(os.path.join(module_path, 'external_data', 'Emoji_Sentiment_Data_v1.0.csv'))
        emoji2 = pd.read_csv(os.path.join(module_path, 'external_data', 'Emojitracker_20150604.csv'))
        emoji = emoji1.merge(emoji2, how='left', on='Emoji', suffixes=('', '_tracker'))
        emoji_list = emoji['Emoji'].values
        
        features = []
        for column in self.text_columns:
            emoji_count = {}
            for e in emoji_list:
                emoji_count[e] = dataframe[column].str.count(e)
            emoji_count = pd.DataFrame(emoji_count)

            emoji_columns = ['Occurrences', 'Position', 'Negative', 'Neutral', 'Positive', 'Occurrences_tracker']
            stats = [np.sum, np.mean, np.max, np.median, np.std]

            feature = {}
            for c in emoji_columns:
                v = emoji_count * emoji[c].values.T
                for stat in stats:
                    feature[column+'_'+stat.__name__+'_'+c] = stat(v, axis=1)
            feature = pd.DataFrame(feature)
            features.append(feature)

        dataframe = pd.concat([dataframe]+features, axis=1)
    
        return dataframe
