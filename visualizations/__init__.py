import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def plot_feature_importances(model, predictors, max_num=50, importance_type='gain', path=None):
    importances = model.feature_importance(importance_type=importance_type)
    importances = pd.DataFrame(np.array([predictors, importances]).T, columns=['feature', 'importances'])
    importances['importances'] = importances['importances'].astype(np.float32)
    importances = importances.sort_values('importances', ascending=False).iloc[:max_num]
    # plot
    if path is not None:
        plt.clf()
        plt.figure(figsize=(16, 6))
        sns.barplot(x='importances', y='feature', data=importances, orient='h')
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
    
    return feature_importances

def plot_prediction_histgram():
    pass
