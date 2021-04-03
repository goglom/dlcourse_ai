import numpy as np


def l2_regularization(W, reg_strength):
    l2_reg_loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return l2_reg_loss, grad

def softmax(predictions):
    copy_predictions = np.copy(predictions)
    if predictions.ndim == 1:
        copy_predictions -= np.max(copy_predictions)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp)
    else:
        copy_predictions -= np.amax(copy_predictions, axis=1, keepdims=True)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp, axis=1, keepdims=True)
    return copy_predictions

def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        loss_func = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        every_batch_loss = -np.log(probs[range(batch_size), target_index])
        loss_func = np.sum(every_batch_loss) / batch_size
    return loss_func


def softmax_with_cross_entropy(preds, target_index):
    d_preds = softmax(preds)
    loss = cross_entropy_loss(d_preds, target_index)
    
    if preds.ndim == 1:
        d_preds[target_index] -= 1
    else:
        batch_size = preds.shape[0]
        d_preds[range(batch_size), target_index] -= 1
        d_preds /= batch_size
    return loss, d_preds


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.where(X >= 0, X, 0.0)

    def backward(self, d_out: np.array) -> np.array:
        d_result = np.where(self.X >= 0.0, 1.0, 0.0) * d_out

        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        self.W.grad += self.X.T @ d_out
        self.B.grad += np.sum(d_out, axis=0)
        d_input = d_out @ self.W.value.T

        return d_input


    def params(self):
        return {'W': self.W, 'B': self.B}
