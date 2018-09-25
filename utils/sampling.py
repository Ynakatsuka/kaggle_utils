import gc
import itertools
import json
import os
import time
import joblib
import numpy as np
import pandas as pd


class DownSampler(object):
    def __init__(self, random_states):
        self.random_states = random_states

    @staticmethod
    def transform(data, target):
        positive_data = data[data[target] == 1]
        positive_ratio = len(positive_data) / len(data)
        negative_data = data[data[target] == 0].sample(
            frac=positive_ratio / (1 - positive_ratio), random_state=self.random_state)
        return positive_data.index.union(negative_data.index).sort_values()
