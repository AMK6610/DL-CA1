import numpy as np 
#import pickle
import function
#import matplotlib.pyplot as plt

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
        # build the network
        #         w1/b1    w2/b2   
        #784(inputs) ---> 20 ---> 10(output)
        #         x     z1  a1  z2  a2=y
        for i in range(len(self.num_nodes_in_layers) - 1):
            self.weights.append(np.random.randn(self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]) * 0.01)
            self.weights.append(np.random.randn(self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]) * 0.01)
        self.loss = []

    def train(self, inputs, labels):

        for epoch in range(self.num_epochs): # training begin
            iteration = 0
            while iteration < len(inputs):

                # batch input
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
                
                #forward pass
                outputs = []
                network_input = inputs_batch
                outputs.append(np.array(inputs_batch))
                # print('network_input :', network_input.shape)
                for i in range(len(self.num_nodes_in_layers) - 1):
                    outputs.append(function.relu(np.dot(network_input, self.weights[i])))
                    network_input = outputs[-1]
                #
                #     # z2 = np.dot(a1, self.weights[1])
                #     # y = function.softmax(z2)
                y = outputs[-1]
                # # calculate loss
                loss = function.cross_entropy(y, labels_batch)
                # loss += function.L2_regularization(0.01, self.weights[0], self.weights[1])#lambda
                self.loss.append(loss)
                delta_y = []
                w_gradient = []
                for i in range(len(self.num_nodes_in_layers) - 2, -1, -1):
                    if i == len(self.num_nodes_in_layers) - 2:
                        delta_y.append(np.array((y - labels_batch) / y.shape[0]))
                    delta_y.append(np.dot(delta_y[-1], self.weights[i].T))
                    index = i
                    # print(index, outputs[1].shape)
                    # print(len(outputs[index]), delta_y[-1].shape)
                    (delta_y[-1])[outputs[index] <= 0] = 0
                    w_gradient.append(np.dot(outputs[index].T, delta_y[-2]))
                    self.weights[index] -= self.learning_rate * w_gradient[-1]

                # # backward pass
                # delta_y = (y - labels_batch) / y.shape[0]
                # delta_hidden_layer = np.dot(delta_y, self.weights[1].T)
                # delta_hidden_layer[outputs[1] <= 0] = 0 # derivatives of relu
                #
                # # backpropagation
                # weight2_gradient = np.dot(outputs[1].T, delta_y) # forward * backward
                # bias2_gradient = np.sum(delta_y, axis = 0, keepdims = True)
                #
                # weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                # bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)
                #
                # # L2 regularization
                # weight2_gradient += 0.01 * self.weights[1]
                # weight1_gradient += 0.01 * self.weights[0]
                #
                # # stochastic gradient descent
                # self.weights[0] -= self.learning_rate * weight1_gradient #update weight and bias
                # self.weights[1] -= self.learning_rate * weight2_gradient

                print('=== Epoch: {:d}/{:d}\tIteration:{:d}\tLoss: {:.2f} ==='.format(epoch+1, self.num_epochs, iteration+1, loss))
                iteration += self.batch_size
        '''
        obj = [self.weight1, self.bias1, self.weight2, self.bias2]
        with open('filename.pkl', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''

    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weights[0])
        hidden_layer = function.relu(input_layer)
        scores = np.dot(hidden_layer, self.weights[1])
        probs = function.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print('Test accuracy: {:.2f}%'.format(acc*100))

