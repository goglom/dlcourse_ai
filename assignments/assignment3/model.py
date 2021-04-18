import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layers = [
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(2 * 2 * conv2_channels, 10)
        ]

    def __clear_grads(self) -> None:
        for param in self.params().values():
            param.grad.fill(0.0)

    def __forward(self, X: np.array) -> np.array:
        layer_output = X

        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        
        return layer_output

    def __backward(self, d_out: np.array) -> None:
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'

        self.__clear_grads()
        forward_out = self.__forward(X)
        loss, d_out = softmax_with_cross_entropy(forward_out, y)
        self.__backward(d_out)

        return loss

    def predict(self, X: np.array):
        forward_out = self.__forward(X)
        pred = np.argmax(forward_out, axis=1)

        return pred

    def params(self):
        result = {
            "W1": self.layers[0].params()['W'],
            "B1": self.layers[0].params()['B'],
            "W2": self.layers[3].params()['W'],
            "B2": self.layers[3].params()['B'],
            "W3": self.layers[7].params()['W'],
            "B3": self.layers[7].params()['B'],
        }
        return result
