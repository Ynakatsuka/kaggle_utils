import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_all(task_type, df, model, true, pred, bins=50, base_path='./', name='', bot=None,
             pred_true_difference_features=None, classes=None, importance_type='gain', importance_path=None, predictors=None):
    to_send = {}
    if (importance_path is not None) and (predictors is not None):
        imp = get_importances_from_model(model, predictors, importance_type=importance_type)
        imp.to_pickle(importance_path)
        plot_feature_importances(imp, max_num=bins, importance_type=importance_type, path=base_path+f'{name}_importance_{importance_type}.png')
        to_send[name+'_'+importance_type] = base_path+f'{name}_importance_{importance_type}.png'

    if task_type in ['binary', 'regression']:
        plot_prediction_histogram(pred, bins=bins, path=base_path+f'{name}_histogram.png')
        plot_true_prediction_histogram(true, pred, path=base_path+f'{name}_true_pred_histogram.png')
        to_send[name+'_'+'histogram'] = base_path+f'{name}_histogram.png'
        if (importance_path is not None) and (predictors is not None):
            pred_true_difference_features += list(imp.sort_values(importance_type, ascending=False)['feature'].iloc[:3])
        for feature in pred_true_difference_features:
            plot_pred_true_difference(df, true, pred, feature, topn=bins, path=base_path+f'{name}_pred_true_difference_per_{feature}.png')
            to_send[name+'_'+f'pred_true_difference_per_{feature}'] = base_path+f'{name}_pred_true_difference_per_{feature}.png'
        plot_lift_chart(true, pred, bins=bins, path=base_path+f'{name}_lift_chart.png')
        to_send[name+'_'+'lift_chart'] = base_path+f'{name}_lift_chart.png'
    elif task_type == 'multiclass':
        plot_confusion_matrix(true, pred, classes, path=base_path+f'{name}_confusion_matrix.png')
        to_send[name+'_'+'lift_chart'] = base_path+f'{name}_lift_chart.png'
    else:
        print('Unsupported task type.')
    if bot is not None:
        for message, path in to_send.items():
            bot.send(message, image=path)


def plot_confusion_matrix(true, pred, classes,
                          normalize=True,
                          cmap=plt.cm.Blues,
                          path=None):
    '''
    Args
    ----
        true: true labels, not one-hot vectors
        pred: results of predict_proba
    '''
    plt.clf()
    # compute cofusion_matrix
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(true, pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.clf()
#     plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
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
    plt.cla(); plt.clf(); plt.close()
    

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

    plt.clf()
#     plt.figure(figsize=(16, 6))
    sns.barplot(x=importance_type, y='feature', data=importances, orient='h')
    plt.title(importance_type)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()


def plot_mean_feature_importances(feature_importances, max_num=50, importance_type='gain', path=None):
    mean_gain = feature_importances[[importance_type, 'feature']].groupby('feature').mean()
    feature_importances['mean_'+importance_type] = feature_importances['feature'].map(mean_gain[importance_type])
    data = feature_importances.sort_values('mean_'+importance_type, ascending=False).iloc[:max_num]

    plt.clf()
#     plt.figure(figsize=(16, 6))
    sns.barplot(x=importance_type, y='feature', data=data)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()


def plot_prediction_histogram(pred, bins=100, path=None):
    plt.clf()
#     plt.figure(figsize=(18, 9))
    plt.hist(pred, bins=bins)
    plt.title('Histogram')
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()
    
    
def plot_true_prediction_histogram(true, pred, path=None):
    df = pd.DataFrame({'true': true, 'pred': pred})
    min_ = min(true.min(), pred.min())
    max_ = max(true.max(), pred.max())
    plt.clf()
#     plt.figure(figsize=(18, 9))
    sns.jointplot('true', 'pred', df, xlim=(min_, max_), ylim=(min_, max_))
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()


def plot_pred_true_difference(dataframe, true, pred, col, topn=50, path=None):
    df = pd.DataFrame({'true': true, 'pred': pred, col: dataframe[col]})
    nunique = df[col].nunique()
    if nunique > topn:
        keys = df[col].value_counts().iloc[:topn].index
        df = df[df[col].isin(keys)]

    plt.clf()
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)    
    ax2 = ax1.twinx()
    
    g = df.groupby(col)
    m = g[['pred', 'true']].mean()
    if isinstance(list(g.groups.keys())[0], str):
        keys = [k for k in g.groups.keys()]
    else:
        keys = [f'{k:.2f}' for k in g.groups.keys()]
    try:
        keys = [float(k) for k in keys]    
    except:
        pass
    
    ax2.bar(keys, g['true'].count(), align='center', color='gray', alpha=0.5)
    ax1.plot(keys, m['pred'], marker='.', markersize=14)
    ax1.plot(keys, m['true'], marker='.', markersize=14)
    ax1.set_xticklabels(keys, rotation=45)
    ax1.set_xlabel(col)
    ax1.set_ylabel('Mean target')
    ax1.grid(True)
    ax2.set_ylabel('Frequency')
    ax2.set_ylim(0, g['true'].count().max()*3)
    plt.title(f'predicted values v.s. true values per {col}')
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()


def plot_lift_chart(true, pred, bins=50, path=None):
    df = pd.DataFrame({'pred': pred, 'true': true}).sort_values('pred')
    chunk_size = np.ceil(len(df) / bins)
    df['bin'] = np.floor(np.arange(1, 1+len(df)) / chunk_size)
    plt.clf()
    df.groupby('bin')[['pred', 'true']].mean().plot(marker='.', markersize=14, figsize=(16, 4))
    plt.grid()
    plt.title('Lift Chart')
    if path is not None:
        plt.savefig(path)
    plt.cla(); plt.clf(); plt.close()
