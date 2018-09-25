import lightgbm as lgb
from copy import deepcopy
from .basic import BaseModel


class LightGBMHelper(BaseModel):
    def __init__(self, data_dir, name, model_params, fit_params, validation_data=(None, None)):
        super(LightGBMHelper, self).__init__(data_dir, name)
        self.model_params = model_params
        self.fit_params = fit_params
        self.validation_data = validation_data
        self.weight_path = self.data_dir + self.name + '.txt'

    def fit(self, X, y):
        train = lgb.Dataset(X,
                            label=y,
                            feature_name=self.model_parms['predictors'],
                            categorical_feature=self.model_parms['categorical'],
                            free_raw_data=False)
        valid = lgb.Dataset(validation_data[0],
                            label=validation_data[1],
                            feature_name=self.predictors,
                            categorical_feature=self.categorical,
                            free_raw_data=False)
        evals_result = {}
        self.model = lgb.train(self.lgb_params,
                             train,
                             valid_sets=[valid],
                             valid_names=['valid'],
                             evals_result=evals_result,
                             num_boost_round=self.num_boost_round,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose_eval)


    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        self.model.save_model(self.weight_path)
