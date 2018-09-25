from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, data_dir, name):
        self.data_dir = data_dir
        self.name = name

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    @staticmethod
    def check_path(path):
        if os.path.exists(path):
            raise ValueError('Weights of this version already exists.')

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
