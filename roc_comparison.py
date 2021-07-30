from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def calc_auc(test_data, probs_data):
    """Calculates the Area under Curve score for the probabilities"""
    return roc_auc_score(test_data, probs_data)

def calc_roc(test_data, probs_data):
    """Calculates the ROC curve for the probabilities"""
    return roc_curve(test_data, probs_data)

def calc_optimal_threshold(fpr, tpr, thresholds):
    """Calculates the optimal threshold for the predictions using Youden’s J statistic 
    (https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)"""
    J = tpr - fpr
    ix = np.argmax(J)
    return thresholds[ix], ix

def plot_roc(test_data, model_probs):
    """Plots the ROC curve for a single model and returns the optimal threshold for probabilities."""
    auc = calc_auc(test_data, model_probs)
    fpr, tpr, thresholds = calc_roc(test_data, model_probs)
    opti_thresh, ix = calc_optimal_threshold(fpr, tpr, thresholds)
    
    print('Model One: ROC AUC=%.3f' % (auc))
    print('Best Model One Threshold=%f' % (opti_thresh))

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='green', label='Best Model One')
    plt.plot(fpr, tpr, label='Model One')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    # show the legend
    plt.legend(loc='best')
    # show the plot
    plt.show()

def compare_roc(test_data, model_one, model_two):
    """Compares the ROC curve and AUC between two models and returns the optimal threshold for probabilities."""
    mo_auc = calc_auc(test_data, model_one)
    mt_auc = calc_auc(test_data, model_two)
    mo_fpr, mo_tpr, mo_thresholds = calc_roc(test_data, model_one)
    mt_fpr, mt_tpr, mt_thresholds = calc_roc(test_data, model_two)
    mo_opti_thresh, mo_ix = calc_optimal_threshold(mo_fpr, mo_tpr, mo_thresholds)
    mt_opti_thresh, mt_ix = calc_optimal_threshold(mt_fpr, mt_tpr, mt_thresholds)


    print('Model One: ROC AUC=%.3f' % (mo_auc))
    print('Model Two: ROC AUC=%.3f' % (mt_auc))

    print('Best Model One Threshold=%f' % (mo_opti_thresh))
    print('Best Model Two Threshold=%f' % (mt_opti_thresh))

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(mo_fpr[mo_ix], mo_tpr[mo_ix], marker='o', color='green', label='Best Model One')
    plt.scatter(mt_fpr[mt_ix], mt_tpr[mt_ix], marker='o', color='red', label='Best Model Two')
    plt.plot(mo_fpr, mo_tpr, label='Model One')
    plt.plot(mt_fpr, mt_tpr, label='Model Two')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    # show the legend
    plt.legend(loc='best')
    # show the plot
    plt.show()
