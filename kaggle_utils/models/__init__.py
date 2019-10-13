import numpy as np
import xgboost as xgb


def predict_by_chunks(model, X, n_chunks=5, categorical_features=[]):
    predictors = X.columns
    chunk_size = -(-len(X) // n_chunks)
    preds = []
    for i in range(n_chunks):
        X_ = X.iloc[i*(chunk_size):min((i+1)*(chunk_size), len(X))]
        if isinstance(model, xgb.Booster):
            X_ = xgb.DMatrix(X_, feature_names=predictors)
        preds.append(model.predict(X_))
    return np.concatenate(preds, axis=0)
