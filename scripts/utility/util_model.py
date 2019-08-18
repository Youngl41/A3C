#======================================================
# Model Utility Functions
#======================================================
'''
Info:       Utility functions for model building.
Version:    2.0
Author:     Young Lee
Created:    Saturday, 13 April 2019
'''
# Import modules
import os
import uuid
import copy
import time
import random 
import numpy as np
import pandas as pd
from subprocess import call
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.utils import resample
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import f1_score


#------------------------------
# Utility Functions
#------------------------------
# Set section
def set_section(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 100
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        full_buffer_len = string_size
        print('\n')
        print(full_buffer_len * '-')
        print(string)
        print(full_buffer_len * '-'+'\n')

def downsample_df(df, labels_df, random_seed):
    num_of_yt           = sum(labels_df)
    random.seed(random_seed+1)
    downsample_bad_ix   = random.sample(np.where(labels_df == 0)[0], num_of_yt)
    good_ix             = np.where(labels_df == 1)[0]
    downsampled_full_ix = np.append(downsample_bad_ix, good_ix)
    df_ds               = pd.concat([df.iloc[[index]] for index in downsampled_full_ix])
    return df_ds

def upsample(df, groupby_cols, random_seed, max_sample_ratio=1.5):
    max_sample_size     = df.groupby(groupby_cols).agg('count').max().max()
    dfs                 = []
    for i, df_ in df.groupby(groupby_cols):
        dfs.append(resample(df_, replace=True, n_samples=int(max_sample_size * max_sample_ratio), random_state=random_seed))
    upsampled_df        = pd.concat(dfs, axis=0)
    return upsampled_df

# Binarise
def binarise_labels(actual, pred):
    classes, actual_pred_binary = np.unique(list(actual.append(pred)), return_inverse = True)
    actual_binary = actual_pred_binary[:len(actual)]
    pred_binary = actual_pred_binary[len(actual):]
    return actual_binary, pred_binary, classes

# Plot confusion
def plot_cfmt(cfmt, classes,
            title='Confusion matrix',
            cmap=plt.cm.Blues,
            save_path=None,
            colorbar=True,
            figsize=(6,6),
            fontsize=None,
            ylabel='True label',
            xlabel='Predicted label'):
    '''
    This function prints and plots the confusion matrix.
    '''
    plt.figure(figsize=figsize)
    plt.imshow(cfmt, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cfmt.max() - (cfmt.max() - cfmt.min())/ 2.
    for i, j in itertools.product(range(cfmt.shape[0]), range(cfmt.shape[1])):
        plt.text(j, i, cfmt[i, j],
                    horizontalalignment="center",
                    color="white" if cfmt[i, j] > thresh else "black", size=fontsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=360)
    else:
        plt.show()

def feature_importance_rf(model, feature_names, verbose=1):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot
    if verbose:
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(feature_names)), importances[indices],
            color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(feature_names)])
        plt.show()

def plot_tree(tree, save_path, feature_names, class_names, dpi=300):
    # Dot path
    dot_save_path = save_path.split('.png')[0] + '.dot'

    # Export as dot file
    export_graphviz(tree, out_file=dot_save_path, 
                    feature_names = feature_names,
                    class_names = class_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    # export_graphviz(tree, out_file=None, max_depth=None, 
    #                 feature_names=None, 
    #                 class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, 
    #                 rotate=False, rounded=False, special_characters=False, precision=3)
    # Convert to png using system command (requires Graphviz)
    
    call(['dot', '-Tpng', dot_save_path, '-o', save_path, '-Gdpi='+str(dpi)])
    os.remove(dot_save_path)

# Classify binary classification predictions
def map_classify_pred(true, pred):
    if (true*1 == 1) & (pred*1 == 1):
        return 'TP'
    elif (true*1 == 0) & (pred*1 == 0):
        return 'TN'
    elif (true*1 == 1) & (pred*1 == 0):
        return 'FN'
    elif (true*1 == 0) & (pred*1 == 1):
        return 'FP'
    else:
        return np.nan

# Clf report
def clf_report(y_series, p_series, verbose=True):
    df = pd.DataFrame(classification_report(y_series, p_series, output_dict=True)).T.reset_index()
    df.columns = ['class', 'f1-score', 'precision', 'recall', 'support']
    agg = df[df['class'].isin(['micro avg', 'macro avg', 'weighted avg'])]
    summary = df[~df['class'].isin(['micro avg', 'macro avg', 'weighted avg'])]
    if verbose:
        set_section('Accuracy: ' + str(np.round(accuracy_score(y_series, p_series),3)))
        print('\n', summary)
        print('\n\n', agg, '\n')
    return summary

def get_acc(y, p, verbose=1, name=None, save_path=None):
    accuracy        = accuracy_score(y, p)
    support         = len(y)
    total_target    = np.sum(y)
    cfmt            = confusion_matrix(y, p)
    precision       = precision_score(y, p, average='weighted')
    recall          = recall_score(y, p, average='weighted')
    f1              = f1_score(y, p, average='weighted')
    acc_breakdown   = clf_report(y_series=y, p_series=p, verbose=False)
    output          = [str(uuid.uuid4()), datetime.now(), name, precision, recall, f1, accuracy, support, total_target, cfmt, [acc_breakdown.to_dict()]]
    output          = pd.DataFrame(output).T
    output.columns  = ['uuid', 'datetime', 'name', 'precision', 'recall', 'f1', 'accuracy', 'support', 'total_target', 'confusion_mat', 'accuracy_breakdown']
    if verbose:
        set_section('Accuracy: ' + str(np.round(accuracy_score(y, p),3)))
        print(acc_breakdown)
    if save_path:
        # Append if file exists
        if os.path.isfile(save_path):
            output.to_csv(save_path, index=False, mode='a', header=False)
        elif not os.path.isfile(save_path):
            output.to_csv(save_path, index=False)
    return output

def gridsearch(X_train, y_train, X_test, y_test, models, params, nfoldCV = 4, score = 'f1', verbose = 1):
    y_train_ = copy.deepcopy(y_train)
    classes, y_train_test_binary    = np.unique(y_train_.append(y_test), return_inverse = True)
    y_train_binary                  = y_train_test_binary[:len(y_train)]
    y_test_binary                   = y_train_test_binary[len(y_train):]


    # Allocate variables
    y_best_preds                    = {}
    best_estimators                 = {}
    
    print ('\n=============================================')
    print ('Tuning hyper-parameters ( ranking: %s' % score, ')')
    print ('============================================='    )
    for model_name in params.keys():
        print ('\n\n--------------------------------')
        print ('Models: %s' % model_name)
        print ('--------------------------------')
        # Define model    
        clf = GridSearchCV(models[model_name], 
                           params[model_name], 
                           cv=nfoldCV,
                           scoring=score,
                           n_jobs = -1,
                           verbose=verbose)
        print ('\n')
        
        # Model train
        clf.fit(X_train, y_train_binary)
        #clf.fit(X_train, y_train)
        
        # Predict on test
        #y_best_pred                 = clf.predict_proba(X_test)
        y_best_pred                 = clf.predict(X_test)
        y_best_preds[model_name]    = y_best_pred
        print ('Done')
        
        # Append best estimators
        best_estimators[model_name] = clf.best_estimator_
    
    # Display results
    print ('\n=============================================')
    print ('Best Hyper-parameters ( ranking: %s' % score, ')')
    print ('============================================='  )
    
    # Score
    print ('\n\n------------------------------')
    print ('Score: %s' % score)
    print ('------------------------------')
        
    # ROC curves
    plt.figure(figsize  = (6,5))
    #y_best_preds_list   = [ v for v in y_best_preds.values()]
    
    # Best classifier
    accuracies                 = []
    for model_name in params.keys():
        #f1                  = f1_score(y_test_binary, y_best_preds[model_name], average = 'macro')
        acc                  = accuracy_score(y_test_binary, y_best_preds[model_name])
        accuracies.append(acc)
    try: 
        best_clf_index      = np.where(accuracies == max(accuracies))[0][0]
    except IndexError:
        best_clf_index = 0
    best_model          = list(params.keys())[best_clf_index]
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print ('Best Model: %20s' % best_model)
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    # Confusion matrix
    cfmt                = confusion_matrix(y_test_binary, y_best_preds[best_model])
    report              = classification_report(y_test_binary, y_best_preds[best_model])

    p_test                  = [classes[i] for i in y_best_preds[best_model]]
    return accuracies, best_model, best_estimators, p_test, cfmt, classes, report