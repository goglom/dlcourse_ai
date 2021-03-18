import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    true_pos_count = np.sum(np.logical_and(prediction, ground_truth))
    correct_count = np.sum(np.logical_not(np.logical_xor(prediction, ground_truth)))
    
    precision = true_pos_count / np.sum(prediction) # true positives + false positives
    recall = true_pos_count / np.sum(ground_truth) # true positives + false negatives
    accuracy = correct_count / prediction.size
    f1 = precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    assert prediction.shape == ground_truth.shape 
    return np.bincount(prediction == ground_truth)[1] / prediction.size
