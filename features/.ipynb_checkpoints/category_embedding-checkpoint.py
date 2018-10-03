import gc
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from .base import BaseFeatureTransformer


class CategoryVectorizer(BaseFeatureTransformer):
    def __init__(self, col1, col2, n_components, 
                 vectorizer=CountVectorizer(), 
                 transformer=LatentDirichletAllocation(),
                 name='CountLDA'):
        self.col1 = col1
        self.col2 = col2
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)

    def transform(self, dataframe):
        sentence = self.create_word_list(dataframe, self.col1, self.col2)
        sentence = self.vectorizer.fit_transform(sentence)
        features = self.transformer.fit_transform(sentence)
        features = self.get_feature(dataframe, self.col1, self.col2, features, name=self.name)
        return pd.concat([dataframe, features], axis=1)

    def create_word_list(self, dataframe, col1, col2):
        col1_size = int(dataframe[col1].values.max() + 1)
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(dataframe[col1].values, dataframe[col2].values):
            col2_list[int(val1)].append(int(val2))
        return [' '.join(map(str, ls)) for ls in col2_list]

    def get_feature(self, dataframe, col1, col2, latent_vector, name=''):
        features = np.zeros(
            shape=(len(dataframe), self.n_components), dtype=np.float32)
        self.columns = ['_'.join([name, col1, col2, str(i)])
                   for i in range(self.n_components)]
        for i, val1 in enumerate(dataframe[col1]):
            features[i, :self.n_components] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=self.columns)

    def get_numerical_features(self):
        return self.columns


class User2Vec(BaseFeatureTransformer):
    def __init__(self, target_feature, user_id_feature, sort_features=None,
                 n_components=300, window=8, min_count=5, workers=4, seed=777, 
                 save_doc2vec_model_path=None, name='user2vec'):
        self.target_feature = target_feature
        self.user_id_feature = user_id_feature
        self.sort_features = sort_features
        self.n_components = n_components
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        self.save_doc2vec_model_path = save_doc2vec_model_path
        self.name = name

    def transform(self, dataframe):
        documents = self.create_documents(dataframe)
        model = Doc2Vec(documents=documents, 
                        vector_size=self.n_components,
                        window=self.window, 
                        min_count=self.min_count, 
                        workers=self.workers, 
                        seed=self.seed)
        if self.save_doc2vec_model_path is not None:
            model.save(self.save_doc2vec_model_path)
        features = dataframe[self.user_id_feature].apply(lambda x: model.docvecs[x])
        features = self.get_feature(features, name=self.name)
        return pd.concat([dataframe, features], axis=1)

    def create_documents(self, dataframe):
        if self.sort_features is not None:
            dataframe = dataframe.sort_values(self.sort_features)
        documents = []
        for user_id in dataframe[self.user_id_feature].unique():
            words = dataframe.loc[dataframe[self.user_id_feature] == user_id, self.target_feature].values
            documents.append(TaggedDocument(words=words, tags=[user_id]))
        if self.sort_features is not None:
            dataframe = dataframe.sort_index()
        return documents

    def get_feature(self, features):
        self.columns = [self.name + '_' + str(i) for i in range(self.n_components)]
        return pd.DataFrame(data=features, columns=self.columns)

    def get_numerical_features(self):
        return self.columns
