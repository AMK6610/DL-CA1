import numpy as np
# import pickle
import function

# import matplotlib.pyplot as plt

num_classes = 10


class Network:

    def __init__(self,
                 num_nodes_in_layers,
                 batch_size,
                 num_epochs,
                 learning_rate,
                 ):
        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights = []
        for i in range(len(self.num_nodes_in_layers) - 1):
            self.weights.append(np.random.randn(self.num_nodes_in_layers[i], self.num_nodes_in_layers[i + 1]) * 0.1)
        self.loss = []

    def train(self, inputs, labels):
        for epoch in range(self.num_epochs):
            iteration = 0
            while iteration < len(inputs):
                inputs_batch = inputs[iteration:iteration + self.batch_size]
                labels_batch = labels[iteration:iteration + self.batch_size]
                outputs = self.forward(inputs_batch)
                y = outputs[-1]

                loss, delta = function.svm_loss(self.weights[-1], outputs[-2], labels_batch)
                # loss = function.cross_entropy(y, labels_batch)
                # loss += function.L2_regularization(0.01, self.weights[0], self.weights[1])#lambda
                self.loss.append(loss)
                labels_batch = np.eye(num_classes)[labels_batch]
                delta_y = []
                w_gradient = []
                delta_y.append(delta)
                for i in range(len(self.num_nodes_in_layers) - 2, -1, -1):
                    delta_y.append(np.dot(delta_y[-1], self.weights[i].T))
                    index = i
                    (delta_y[-1])[outputs[index] <= 0] = 0
                    w_gradient.append(np.dot(outputs[index].T, delta_y[-2]))
                    self.weights[index] -= self.learning_rate * w_gradient[-1]

                # delta_hidden_layer = np.dot(delta_y, self.weights[1].T)
                # delta_hidden_layer[outputs[1] <= 0] = 0  # derivatives of relu
                #
                # # backpropagation
                # weight2_gradient = np.dot(outputs[1].T, delta_y)  # forward * backward
                #
                # weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                #
                # # L2 regularization
                # weight2_gradient += 0.01 * self.weights[1]
                # weight1_gradient += 0.01 * self.weights[0]
                #
                # # stochastic gradient descent
                # self.weights[0] -= self.learning_rate * weight1_gradient  # update weight and bias
                # self.weights[1] -= self.learning_rate * weight2_gradient

                print('=== Epoch: {:d}/{:d}\tIteration:{:d}\tLoss: {:.2f} ==='.format(epoch + 1, self.num_epochs,
                                                                                      iteration + 1, loss))
                iteration += self.batch_size
        '''
        obj = [self.weights[0], self.bias1, self.weights[1], self.bias2]
        with open('filename.pkl', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''

    def forward(self, inputs):
        outputs = []
        network_input = inputs
        outputs.append(inputs)
        for i in range(len(self.num_nodes_in_layers) - 1):
            outputs.append(function.relu(np.dot(network_input, self.weights[i])))
            network_input = outputs[-1]
        return outputs

    def test(self, inputs, labels):
        outputs = self.forward(inputs)
        acc = float(np.sum(np.argmax(outputs[-1], 1) == labels)) / float(len(labels))
        print('Test accuracy: {:.2f}%'.format(acc * 100))
