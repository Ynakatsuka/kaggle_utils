import networkx as nx
from node2vec import Node2Vec

from .base import BaseFeatureTransformer


class GraphVectorizer(BaseFeatureTransformer):
    '''
    Reference
    ---------
    https://github.com/senkin13/kaggle/blob/master/elo/graph.py
    
    Example
    -------
    vectorizer = GraphVectorizer(
        ['card_id','merchant_id'], 
        n_components=64, walk_length=30, num_walks=100, workers=4, 
        window=5, min_count=1, batch_words=4, name='node2vec'
    )
    new_transactions = vectorizer.fit_transform(new_transactions)
    '''
    def __init__(
        self, categorical_columns, 
        n_components=64, walk_length=30, num_walks=100, workers=4, 
        window=5, min_count=1, batch_words=4, 
        name='node2vec'
    ):
        self.categorical_columns = categorical_columns
        assert len(categorical_columns) == 2
        self.n_components = n_components
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.name = name
        
    def fit(self, dataframe):
        edges = dataframe.groupby(
            self.categorical_columns, 
            as_index=False
        ).size().reset_index().dropna()
        
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges.values)

        node2vec = Node2Vec(
            G, 
            dimensions=self.n_components, 
            walk_length=self.walk_length, 
            num_walks=self.num_walks, 
            workers=self.workers,
        )
        self.model = node2vec.fit(
            window=self.window, 
            min_count=self.min_count, 
            batch_words=self.batch_words,
        )
        self.feature = pd.DataFrame({
            key: self.model.wv[key] for key in self.model.wv.vocab
        }).T.reset_index()
        self.feature.columns = self.categorical_columns[:1] + [
            f'{self.name}_{i:03}' for i in range(self.n_components)
        ]
        self.features = [self.feature]
        return self
    
    def transform(self, dataframe):
        dataframe = dataframe.merge(self.feature, on=self.categorical_columns[:1], how='left')
        return dataframe
    
    def fit_transform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)
