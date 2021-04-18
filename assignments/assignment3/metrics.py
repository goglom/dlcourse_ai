import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    true_pos_count = np.sum(np.logical_and(prediction, ground_truth))
    correct_count = np.sum(np.logical_not(np.logical_xor(prediction, ground_truth)))
    
    precision = true_pos_count / np.sum(prediction) # true positives + false positives
    recall = true_pos_count / np.sum(ground_truth) # true positives + false negatives
    accuracy = correct_count / prediction.size
    f1 = precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]
    
    return accuracy