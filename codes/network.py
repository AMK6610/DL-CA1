import copy

import numpy as np
import pickle
import function

import matplotlib.pyplot as plt

num_classes = 10
l2_r = 0.0001
weights_file = 'weights.pkl'


class Network:

    def __init__(self,
                 num_nodes_in_layers,
                 batch_size,
                 num_epochs,
                 learning_rate,
                 momentum,
                 weights_file,
                 activation='relu',
                 # activation='softplus',
                 # activation='lrelu'
                 ):
        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights_file = weights_file
        self.weights = []
        sd = 0.01
        for i in range(len(self.num_nodes_in_layers) - 1):
            self.weights.append(np.random.randn(self.num_nodes_in_layers[i], self.num_nodes_in_layers[i + 1]))
        self.loss = []
        self.test_loss = []
        self.accuracy = []
        self.activation = activation

    def apply_activation_derivative(self, input, delta):
        if self.activation == 'relu':
            return function.d_relu(input, delta)
        if self.activation == 'softplus':
            return function.d_softplus(input, delta)
        if self.activation == 'lrelu':
            return function.d_lrelu(input, delta)

    def apply_activation(self, input):
        if self.activation == 'relu':
            return function.relu(input)
        if self.activation == 'softplus':
            return function.softplus(input)
        if self.activation == 'lrelu':
            return function.lrelu(input)

    def train(self, inputs, labels, test_inputs, test_labels):
        print("Training...")
        if self.weights_file is not None:
            file = pickle.load(open(self.weights_file, "rb"))
            self.weights = file['weights']
            self.loss = file['loss']
            self.accuracy = file['accuracy']
            print('reading weights from {:s}...'.format(self.weights_file))
            # self.draw_figs()
            return
        prev_w_gradient = []
        for epoch in range(self.num_epochs):
            acc, l = self.test(test_inputs, test_labels)
            self.accuracy.append(acc)
            self.test_loss.append(l)
            iteration = 0
            iteration_loss = []

            while iteration < len(inputs):
                inputs_batch = inputs[iteration:iteration + self.batch_size]
                labels_batch = labels[iteration:iteration + self.batch_size]
                outputs, res = self.forward(inputs_batch, labels_batch)
                loss = res[0]
                delta = res[1]
                # loss += function.L2_regularization(l2_r, self.weights[0], self.weights[1]) # L2 Regularization
                iteration_loss.append(loss)
                # labels_batch = np.eye(num_classes)[labels_batch]
                delta_y = []
                w_gradient = []
                delta_y.append(delta)
                for i in range(len(self.num_nodes_in_layers) - 2, -1, -1):
                    delta_y.append(np.dot(delta_y[-1], self.weights[i].T))
                    index = len(self.num_nodes_in_layers) - 2 - i
                    delta_y[-1] = self.apply_activation_derivative(outputs[i], delta_y[-1])
                    w_gradient.append(np.dot(outputs[i].T, delta_y[-2]))
                    if epoch == 0:
                        self.weights[i] -= self.learning_rate * w_gradient[-1] #- l2_r * self.weights[i] # L2 Regularization
                    else:
                        self.weights[i] -= self.learning_rate * (w_gradient[-1] + self.momentum * prev_w_gradient[index])
                prev_w_gradient = copy.deepcopy(w_gradient)

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
            self.loss.append(np.mean(np.array(iteration_loss)))
        obj = {'weights': self.weights, 'loss': self.loss, 'accuracy': self.accuracy}
        pickle.dump(obj, open(weights_file, "wb"))
        # self.draw_figs()

    def forward(self, inputs, labels):
        outputs = []
        network_input = inputs
        outputs.append(inputs)
        for i in range(len(self.num_nodes_in_layers) - 1):
            outputs.append(self.apply_activation(np.dot(network_input, self.weights[i])))
            network_input = outputs[-1]
        return outputs, function.svm_loss(self.weights[-1], outputs[-2], labels)

    def test(self, inputs, labels):
        outputs, res = self.forward(inputs, labels)
        loss = res[0]
        acc = float(np.sum(np.argmax(outputs[-1], 1) == labels)) / float(len(labels))
        print('Test accuracy: {:.2f}%'.format(acc * 100))
        return acc * 100, loss

    def draw_figs(self):
        plt.plot([i + 1 for i in range(self.num_epochs)], self.loss, marker='o')
        # plt.plot([i + 1 for i in range(self.num_epochs)], self.test_loss, marker='o')
        plt.title('Changes in loss function')
        plt.xlabel('Epoch')
        plt.ylabel('Hinge')
        plt.figure()
        plt.plot([i + 1 for i in range(self.num_epochs)], self.accuracy, marker='o', color='red')
        plt.title('Changes in accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.figure()
        plt.show()
