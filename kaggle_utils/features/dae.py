from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, QuantileTransformer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence

from .base import BaseFeatureTransformer


def on_field(f, *vec):
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


class DAEDataLoader(Sequence):
    def __init__(self, x, batch_size=128, swap_rate=0.15):
        self.x = x
        self.batch_size = batch_size
        self.swap_rate = swap_rate
        self.random_idx = np.random.permutation(len(x))
        self.n_features = x.shape[1]
        self.n_swap = int(swap_rate * self.n_features)

    def __len__(self):
        return -(-len(self.x) // self.batch_size)

    def __getitem__(self, idx):
        x_batch = self.x[idx*self.batch_size: (idx + 1)*self.batch_size]
        x_random_batch = self.x[self.random_idx[idx*self.batch_size: (idx + 1)*self.batch_size]]
        new_batch = x_batch.copy()
        
        for i in range(len(x_batch)):
            swap_idx = np.random.choice(self.n_features, self.n_swap, replace=False)
            new_batch[i, swap_idx] = x_random_batch[i, swap_idx]

        return new_batch, x_batch


class DAETransformer(BaseFeatureTransformer):
    def __init__(
        self, categorical_features, numerical_features,
        numerical_preprocessor=QuantileTransformer(),
        swap_rate=0.15, batch_size=128, epochs=100, 
        optimizer=optimizers.Adam(),
        loss='mse', metrics=['mse'],
        n_units=500, n_layers=3, name='dae'
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.numerical_preprocessor = numerical_preprocessor
        self.swap_rate = swap_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.n_units = n_units
        self.n_layers = n_layers
        self.name = name
        
    def _get_model(self, input_dim):
        inputs = Input((input_dim,))
        x = Dense(self.n_units, activation='relu', name='layer1')(inputs)
        for _ in range(self.n_layers-1):
            x = Dense(self.n_units, activation='relu', name=f'layer{_+2}')(x)
        outputs = Dense(input_dim)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=self.loss, 
            metrics=self.metrics, 
            optimizer=self.optimizer,
        )
        return model
    
    def _get_middle_model(self):
        outputs = [layer.output for layer in self.model.layers[1:-1]]
        model = Model(inputs=self.model.layers[0].input, outputs=outputs)
        return model
        
    def get_features(self, X):
        features = np.hstack(self.middle_model.predict(X))
        columns = [f'{self.name}_{i:03}' for i in range(features.shape[1])]
        return pd.DataFrame(features, columns=columns)
    
    def fit(self, dataframe):        
        self.preprocessor = make_union(
            on_field(
                self.categorical_features, 
                OneHotEncoder(categories='auto', sparse=False, dtype=np.float32)
            ),
            on_field(self.numerical_features, self.numerical_preprocessor),
        )
        X = self.preprocessor.fit_transform(dataframe)
        
        self.model = self._get_model(X.shape[1])
        loader = DAEDataLoader(X, self.batch_size, self.swap_rate)
        self.model.fit(
            loader, 
            epochs=self.epochs,
        )
        self.middle_model = self._get_middle_model()
        return self
    
    def transform(self, dataframe):
        X = self.preprocessor.transform(dataframe)        
        features = self.get_features(X)
        return pd.concat([dataframe, features], axis=1)
    
    def fit_transform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)
