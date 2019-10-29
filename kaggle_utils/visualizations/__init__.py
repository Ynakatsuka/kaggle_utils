import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true, pred, classes,
                          normalize=True,
                          title='Confusion matrix',
                          path=None,
                          cmap=plt.cm.Blues):
    '''
    Args
    ----
        true: true labels, not one-hot vectors
        pred: results of predict_proba
    '''
    # compute cofusion_matrix
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(true, pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.clf()
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)


def get_importances_from_model(model, predictors, importance_type='gain'):
    # lightgbm
    if hasattr(model, 'feature_importance'):
        importances = model.feature_importance(importance_type=importance_type)
    # catboost
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    # xgboost
    elif hasattr(model, 'get_score'):
        if importance_type == 'split':
            importances = model.get_score(importance_type='weight')
        elif importance_type == 'gain':
            importances = model.get_score(importance_type='total_gain')
        else:
            importances = model.get_score(importance_type=importance_type)
        predictors = list(importances.keys())
        importances = list(importances.values())
    # sklearn warpper and catboost
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_(importance_type=importance_type)
    
    result = pd.DataFrame({
        'feature': predictors,
        importance_type: importances
    })
    
    return result
    

def plot_feature_importances(importances, max_num=50, importance_type='gain', path=None):    
    importances = importances.sort_values(importance_type, ascending=False).iloc[:max_num]
    # plot
    if path is not None:
        plt.clf()
        plt.figure(figsize=(16, 6))
        sns.barplot(x=importance_type, y='feature', data=importances, orient='h')
        plt.title(importance_type)
        plt.tight_layout()
        plt.savefig(path)
        

def plot_mean_feature_importances(feature_importances, max_num=50, importance_type='gain', path=None):
    mean_gain = feature_importances[[importance_type, 'feature']].groupby('feature').mean()
    feature_importances['mean_'+importance_type] = feature_importances['feature'].map(mean_gain[importance_type])

    if path is not None:
        data = feature_importances.sort_values('mean_'+importance_type, ascending=False).iloc[:max_num]
        plt.clf()
        plt.figure(figsize=(16, 6))
        sns.barplot(x=importance_type, y='feature', data=data)
        plt.tight_layout()
        plt.savefig(path)


def plot_prediction_histogram(y, bins=100, path=None):
    if path is not None:
        plt.clf()
        plt.figure(figsize=(12, 6))
        plt.hist(y, bins=bins)
        plt.title('Histogram')
        plt.savefig(path)


def plot_actual_difference(df, pred, target, col, topn=50, path=None):
    train = df[[col, target]].copy()
    train['__pred'] = pred
    nunique = train[col].nunique()
    if nunique > topn:
        keys = train[col].value_counts().iloc[:topn].index
        train = train[train[col].isin(keys)]

    plt.clf()
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(111)    
    ax2 = ax1.twinx()
    
    g = train.groupby(col)
    m = g[['__pred', target]].mean()
    if isinstance(list(g.groups.keys())[0], str):
        keys = [k for k in g.groups.keys()]
    else:
        keys = [f'{k:.2f}' for k in g.groups.keys()]
    ax2.bar(keys, g[target].count(), align='center', color='gray', alpha=0.5)
    ax1.plot(keys, m['__pred'], marker='.', markersize=14)
    ax1.plot(keys, m[target], marker='.', markersize=14)
    ax1.set_xticklabels(keys, rotation=45)
    ax1.set_xlabel(col)
    ax1.set_ylabel('Mean target')
    ax1.grid(True)
    ax2.set_ylabel('Frequency')
    ax2.set_ylim(0, g[target].count().max()*3)
    if path is not None:
        plt.savefig(path)


def plot_lift_chart(true, pred, bins=50, path=None):
    plt.clf()
    df = pd.DataFrame({'pred': pred, 'true': true}).sort_values('pred')
    chunk_size = np.ceil(len(df) / bins)
    df['bin'] = np.floor(np.arange(1, 1+len(df)) / chunk_size)
    df.groupby('bin')[['pred', 'true']].mean().plot(marker='.', markersize=14, figsize=(16, 4))
    plt.grid()
    plt.title('Lift Chart')
    if path is not None:
        plt.savefig(path)
        