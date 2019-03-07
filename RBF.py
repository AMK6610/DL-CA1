from NN import NeuralNetwork, Layer
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

hidden_neurons = 10
out_neurons = 10
learning_rate = 1e-14
epoch = 1000
batch_size = 128

num_train = 60000

np.random.seed(100)


class RBFLayer(Layer):
    def __init__(self, n_input, n_neurons, layer, activation=None, weights=None, centers=None, sigma=None):
        super().__init__(n_input, n_neurons, activation, weights)
        self.centers = centers if centers is not None else np.zeros((n_neurons, n_input))
        self.sigma = sigma if sigma is not None else np.ones(n_neurons)
        self.layer = layer

    def activate(self, x):
        if self.layer == 'hidden':
            tmp = []
            for i in range(self.centers.shape[0]):
                tmp.append(np.exp(-(np.power(np.subtract(x, self.centers[i]), 2) / (2 * np.power(self.sigma[i], 2)))))
            self.last_activation = np.array(tmp)
        elif self.layer == 'output':
            self.last_activation = np.dot(x.T, self.weights)
        # print(self.last_activation.shape)
        return self.last_activation


class RBF(NeuralNetwork):
    def backpropagation(self, X, y, learning_rate):
        mse = np.mean(np.square(y - self.feed_forward(X)))
        print('Epoch: MSE: %f' % (float(mse)))
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
                layer.delta = np.sum(output - y, axis=1)
                input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
                layer.weights -= np.sum(np.dot(np.moveaxis(input_to_use, 0, -2), layer.delta), axis=0) * learning_rate
                # The output = layer.last_activation in this case

                # print('layer error', i, layer.delta.shape)
            else:
                next_layer = self._layers[i + 1]
                # print('shapes', i,  next_layer.weights.shape, next_layer.delta.shape)
                layer.error = np.sum(np.dot(next_layer.delta, next_layer.weights.T))
                # for i in range(X.shape[0]):
                #     for j in range(layer.centers.shape[0]):
                #         second_part = (X[i] - layer.centers[j][i]) / np.power(layer.sigma[j], 2)
                for j in range(layer.centers.shape[0]):
                    second_part = np.subtract(X, layer.centers[j]) / np.power(layer.sigma[j], 2)
                        # print(second_part.shape)
                # print(np.sum(learning_rate * layer.last_activation * second_part * layer.error, axis=1).shape)
                layer.centers -= np.sum(learning_rate * layer.last_activation * second_part * layer.error, axis=1)
                # for i in range(len(layer.centers)):
                #     dist = np.subtract(X - layer.centers[i])
                # layer.delta = np.dot(layer.error, np.moveaxis(layer.last_activation, 0, -1)) * (dist / (layer.sigma ** 2))
                # layer.centers -= learning_rate * layer.delta


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
    nn = RBF()
    nn.add_layer(RBFLayer(x_train.shape[1], hidden_neurons, layer='hidden'))
    nn.add_layer(RBFLayer(hidden_neurons, out_neurons, layer='output'))
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
