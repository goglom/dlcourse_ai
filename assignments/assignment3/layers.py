import numpy as np
from itertools import product

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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding):
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size, in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = height + 2 * self.padding - self.filter_size + 1
        out_width = width + 2 * self.padding - self.filter_size + 1
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        matW = np.reshape(self.W.value, 
                    (self.filter_size**2 * self.in_channels, self.out_channels))
        paddedX = np.zeros((batch_size, height + 2* self.padding, width + 2*self.padding, channels))
        
        if (self.padding == 0):
            paddedX = X
        else:
            paddedX[:, self.padding: -self.padding, self.padding: -self.padding, :] = X

        self.X = paddedX
        new_shape = (batch_size, self.filter_size**2 * self.in_channels)

        for y, x in product(range(out_height), range(out_width)):
            window = paddedX[:, y: y + self.filter_size, x: x + self.filter_size, :]
            matX = np.reshape(window, new_shape)
            output[:, y, x, :] = matX @ matW + self.B.value
        
        return output


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_inp = np.zeros_like(self.X)
        newXShape = (batch_size, self.filter_size**2 * self.in_channels)
        matW = np.reshape(self.W.value, 
                    (self.filter_size**2 * self.in_channels, self.out_channels)) 

        for y, x in product(range(out_height), range(out_width)):
            window = self.X[:, y: y+self.filter_size, x: x+self.filter_size, :]
            matX = np.reshape(window, newXShape)

            self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
            mat_dW = matX.T @ d_out[:, y, x, :]
            self.W.grad += mat_dW.reshape(self.W.grad.shape)

            dX = d_out[:, y, x, :] @ matW.T
            dX = dX.reshape((batch_size, self.filter_size, self.filter_size, self.in_channels))

            d_inp[:, y: y+self.filter_size, x: x+self.filter_size, :] += dX

        if (self.padding == 0):
            return d_inp

        return d_inp[:, self.padding: -self.padding, self.padding: -self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output_shape = (batch_size, out_height, out_width, channels)
        output = np.zeros(output_shape)
        
        for y, x in product(range(out_height), range(out_width)):
            h_beg, w_beg = y * self.stride, x * self.stride
            h_end, w_end = h_beg + self.pool_size, w_beg + self.pool_size
            
            I = X[:, h_beg:h_end, w_beg:w_end, :]
            self.build_mask(x=I, pos=(y, x))
            output[:, y, x, :] = np.max(I, axis=(1, 2))
            
        return output
        

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(self.X)
        
        for y, x in product(range(out_height), range(out_width)):
            h_beg, w_beg = y * self.stride, x * self.stride
            h_end, w_end = h_beg + self.pool_size, w_beg + self.pool_size
            
            dX[:, h_beg:h_end, w_beg:w_end, :] += d_out[:, y:y+1, x:x+1, :] * self.masks[(y, x)]   
        return dX
    
    def build_mask(self, x, pos):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.masks[pos] = mask

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
