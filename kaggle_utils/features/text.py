import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow_hub as hub
from transformers import pipeline

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

    
class W2VFeatureTransformer(BaseFeatureTransformer):
    '''
    from gensim.models import FastText, word2vec, KeyedVectors
    
    model = word2vec.Word2Vec.load('../data/w2v.model')
    # model = KeyedVectors.load_word2vec_format(path, binary=True)
    '''
    def __init__(self, text_columns, model, name='w2v'):
        self.text_columns = text_columns
        self.model = model
        self.name = name
        
    def transform(self, dataframe):
        self.features = []
        for c in self.text_columns:
            texts = dataframe[c].astype(str)
            result = []
            for text in texts:
                n_skip = 0
                vec = np.zeros(self.model.vector_size)
                for n_w, word in enumerate(text):
                    if self.model.__contains__(word):
                        vec = vec + self.model[word]
                vec = vec / (n_w - n_skip + 1)
                result.append(vec)
            result = pd.DataFrame(
                result, 
                columns=[f'{c}_{self.name}_{i:03}' for i in range(self.model.vector_size)]
            )
            self.features.append(result)
        dataframe = pd.concat([dataframe]+self.features, axis=1)
        return dataframe


class USEFeatureTransformer(BaseFeatureTransformer):
    '''
    Example
    -------
    urls = [
        'https://tfhub.dev/google/universal-sentence-encoder/4',
    ]
    '''
    def __init__(self, text_columns, urls, name='use'):
        self.text_columns = text_columns
        self.urls = urls
        self.name = name
        
    def transform(self, dataframe):
        self.features = []
        for url in self.urls: 
            model_name = url.split('/')[-2]
            embed = hub.load(url)
            for c in self.text_columns:
                texts = dataframe[c].astype(str)
                result = embed(texts).numpy()
                result = pd.DataFrame(
                    result, 
                    columns=[f'{self.name}_{model_name}_{i:03}' for i in range(result.shape[1])]
                )
                self.features.append(result)
        dataframe = pd.concat([dataframe]+self.features, axis=1)
        return dataframe

    
class BERTFeatureTransformer(BaseFeatureTransformer):
    '''
    Reference
    ---------
    https://huggingface.co/transformers/pretrained_models.html
    
    Example
    -------
    '''
    def __init__(self, text_columns, model_names, batch_size=8, device=-1):
        self.text_columns = text_columns
        self.model_names = model_names
        self.batch_size = batch_size
        self.device = device

    def transform(self, dataframe):
        self.features = []
        for model_name in self.model_names: 
            model = pipeline('feature-extraction', device=self.device, model=model_name)            
            for c in self.text_columns:
                texts = dataframe[c].astype(str).tolist()
                result = []
                for i in range(np.ceil(len(texts)/self.batch_size).astype(int)):
                    result.append(
                        np.max(model(
                            texts[i*self.batch_size:min(len(texts), (i+1)*self.batch_size)]
                        ), axis=1)
                    )
                result = np.concatenate(result, axis=0)
                result = pd.DataFrame(
                    result, 
                    columns=[f'{model_name}_{i:03}' for i in range(result.shape[1])]
                )
                self.features.append(result)
        dataframe = pd.concat([dataframe]+self.features, axis=1)
        return dataframe


class BM25Transformer(BaseEstimator, TransformerMixin):
    '''
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)

    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    '''
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        '''
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        '''
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        '''
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        '''
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X
