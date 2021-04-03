import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]


    def __zeros_grad(self) -> None:
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

    def __regularization(self) -> float:
        total_reg_los = 0.0
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            param.grad += reg_grad
            total_reg_los += reg_loss
        
        return total_reg_los

    def compute_loss_and_gradients(self, X: np.array, y: np.array):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'

        self.__zeros_grad()
        forward_out = self.__forward(X)
        loss, d_out = softmax_with_cross_entropy(forward_out, y)
        self.__backward(d_out)
        loss += self.__regularization()

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        forward_out = self.__forward(X)
        pred = np.argmax(forward_out, axis=1)

        return pred

    def params(self):
        result = {
            "W1": self.layers[0].params()['W'],
            "B1": self.layers[0].params()['B'],
            "W2": self.layers[2].params()['W'],
            "B2": self.layers[2].params()['B'],
        }

        return result
