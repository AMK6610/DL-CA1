import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

hidden_neurons = 200
out_neurons = 10
learning_rate = 1e-12
epoch = 5000
batch_size = 128

num_train = 60000

np.random.seed(100)

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """

        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons) * 0.01
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.01
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # In case no activation function was chosen
        if self.activation is None:
            return r

        # tanh
        if self.activation == 'tanh':
            return np.tanh(r)

        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        if self.activation == 'relu':
            return r * (r > 0)

        return r

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        if self.activation == 'relu':
            return 1. * (r > 0)

        return r


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            X = layer.activate(X)

        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)

        # One row
        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows
        print(ff[0], np.argmax(ff, axis=1)[0])
        return np.argmax(ff, axis=1)

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X)
        # print('output:', output)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = output - y
                # The output = layer.last_activation in this case
                layer.delta = np.multiply(layer.error, layer.apply_activation_derivative(output))
                # print('layer error', i, layer.delta.shape)
            else:
                next_layer = self._layers[i + 1]
                # print('shapes', i,  next_layer.weights.shape, next_layer.delta.shape)
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                layer.delta = np.multiply(layer.error, layer.apply_activation_derivative(layer.last_activation))
                # print('layer error', i, layer.delta.shape)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            # print(input_to_use[input_to_use != 0])
            # print(layer.delta)
            # print('layer', i, 'weights:', input_to_use.T)
            # print(np.dot(input_to_use.T, layer.delta) * learning_rate)
            layer.weights -= np.dot(input_to_use.T, layer.delta) * learning_rate

    def train(self, X, y, learning_rate, max_epochs):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        mses = []

        for i in range(max_epochs):
            # for j in range(len(X)):
                # print('hi')
            for j in range(int(X.shape[0]/batch_size)):
                self.backpropagation(X[j*batch_size: (j+1)*batch_size], y[j*batch_size: (j+1)*batch_size], learning_rate)
            self.backpropagation(X[j*batch_size: X.shape[0]], y[j*batch_size: y.shape[0]], learning_rate)
            # if i % 10 == 0:
            mse = np.mean(np.square(y - self.feed_forward(X)))
            mses.append(mse)
            print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses

    @staticmethod
    def accuracy(y_pred, y_true):
        print(y_pred.shape, y_true.shape)
        print(y_pred, y_true)
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """

        return (y_pred == y_true).mean()


def read_data():
    x_train, y_train = loadlocal_mnist(
        images_path='data/mnist/train-images.idx3-ubyte',
        labels_path='data/mnist/train-labels.idx1-ubyte')
    x_test, y_test = loadlocal_mnist(
        images_path='data/mnist/t10k-images.idx3-ubyte',
        labels_path='data/mnist/t10k-labels.idx1-ubyte')
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = read_data()
    y_train = np.array([[0 if y != i else 1 for i in range(10)] for y in y_train])
    # y_test2 = y_test
    # y_test = np.array([[0 if y != i else 1 for i in range(10)] for y in y_test])
    nn = NeuralNetwork()
    nn.add_layer(Layer(x_train.shape[1], hidden_neurons, 'relu'))
    nn.add_layer(Layer(hidden_neurons, out_neurons, 'relu'))
    errors = nn.train(x_train, y_train, learning_rate, epoch)
    print('Accuracy: %.2f%%' % (nn.accuracy(nn.predict(x_test), y_test) * 100))

    # Plot changes in mse
    plt.plot(errors)
    plt.title('Changes in MSE')
    plt.xlabel('Epoch (every 10th)')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    main()
