import itertools
import pickle

import numpy as np
from network import Network
import function
from mpl_toolkits import mplot3d
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


def read_data():
    x_train, y_train = loadlocal_mnist(
        images_path='../data/mnist/train-images.idx3-ubyte',
        labels_path='../data/mnist/train-labels.idx1-ubyte')
    x_test, y_test = loadlocal_mnist(
        images_path='../data/mnist/t10k-images.idx3-ubyte',
        labels_path='../data/mnist/t10k-labels.idx1-ubyte')
    return x_train, y_train, x_test, y_test


num_classes = 10
n_components = 784
x_dim_div = 5
y_dim_div = 5


def main():
    x_train, y_train, x_test, y_test = read_data()
    x_train = x_train / 255
    x_test = x_test / 255
    for momentum in [0, 0.5, 0.9, 0.99]:
        accuracy = []
        for lr in list(np.linspace(0.0001, 0.01, y_dim_div)):
            print('learning with lr', lr)
            for b_size in list(np.linspace(32, 512, x_dim_div)):
                print('learning with batch size', b_size)
                net = Network(
                    num_nodes_in_layers=[n_components, 200, 10],
                    batch_size=int(b_size),
                    num_epochs=10,
                    learning_rate=lr,
                    momentum=momentum,
                    # weights_file="weights.pkl",
                    # weights_file="weights_87acc_20ep_noreg_softplus.pkl"
                    weights_file=None,
                )
                net.train(x_train, y_train, x_test, y_test)
                print("Testing...")
                acc, loss = net.test(x_test, y_test)
                accuracy.append(acc)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x_dim = list(np.linspace(32, 512, x_dim_div)) * y_dim_div
        y_dim = list(np.linspace(0.0001, 0.01, y_dim_div))
        y_dim = list(itertools.chain.from_iterable(itertools.repeat(x, x_dim_div) for x in y_dim))
        ax.scatter3D(x_dim, y_dim, accuracy)
        plt.title('Changes in accuracy function with momentum=' + str(momentum))
        plt.xlabel('Batch Size')
        plt.ylabel('Learning Rate')
        plt.show()
        plt.savefig('../img/3d_acc_' + str(momentum) + '.png')

    #
    # net = Network(
    #     num_nodes_in_layers=[n_components, 200, 10],
    #     batch_size=128,
    #     num_epochs=500,
    #     learning_rate=0.01,
    #     momentum=0.9,
    #     # weights_file="weights.pkl",
    #     # weights_file="weights_87acc_20ep_noreg_softplus.pkl"
    #     weights_file=None,
    # )
    #
    # x_train = function.PCA(x_train, n_components)
    # x_test = function.PCA(x_test, n_components)
    # # x_train = function.normalize(x_train)
    # # x_test = function.normalize(x_test)
    #
    # net.train(x_train, y_train, x_test, y_test)
    #
    # print("Testing...")
    # net.test(x_test, y_test)

    # file = pickle.load(open('weights.pkl', "rb"))
    # weights = file['weights']
    # self.draw_figs()


if __name__ == '__main__':
    main()
