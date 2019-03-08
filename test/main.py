# import numpy as np
# from network import Network
# import mnist
# from mlxtend.data import loadlocal_mnist
#
# def read_data():
#     x_train, y_train = loadlocal_mnist(
#         images_path='../data/mnist/train-images.idx3-ubyte',
#         labels_path='../data/mnist/train-labels.idx1-ubyte')
#     x_test, y_test = loadlocal_mnist(
#         images_path='../data/mnist/t10k-images.idx3-ubyte',
#         labels_path='../data/mnist/t10k-labels.idx1-ubyte')
#     return x_train, y_train, x_test, y_test
#
# # load data
# num_classes = 10
# # train_images = mnist.train_images() #[60000, 28, 28]
# # train_labels = mnist.train_labels()
# # test_images = mnist.test_images()
# # test_labels = mnist.test_labels()
#
# print("Training...")
#
# # data processing
# # X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
# # x_train = X_train / 255 #normalization
# # y_train = np.eye(num_classes)[train_labels] #convert label to one-hot
# #
# # X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
# # x_test = X_test / 255 #normalization
# # y_test = test_labels
# x_train, y_train, x_test, y_test = read_data()
# y_train = np.eye(num_classes)[y_train]
#
#
# net = Network(
#                  num_nodes_in_layers = [784, 200, 10],
#                  batch_size = 128,
#                  num_epochs = 5,
#                  learning_rate = 0.001,
#              )
#
# net.train(x_train, y_train)
#
#
# print("Testing...")
# net.test(x_test, y_test)



import pickle

import numpy as np
from network import Network
import mnist
import function
from mlxtend.data import loadlocal_mnist


def read_data():
    x_train, y_train = loadlocal_mnist(
        images_path='../data/mnist/train-images.idx3-ubyte',
        labels_path='../data/mnist/train-labels.idx1-ubyte')
    x_test, y_test = loadlocal_mnist(
        images_path='../data/mnist/t10k-images.idx3-ubyte',
        labels_path='../data/mnist/t10k-labels.idx1-ubyte')
    return x_train, y_train, x_test, y_test


num_classes = 10
n_components = 64


def main():
    x_train, y_train, x_test, y_test = read_data()
    x_train = x_train / 255
    x_test = x_test / 255
    net = Network(
        num_nodes_in_layers=[n_components, 200, 10],
        batch_size=128,
        num_epochs=500,
        learning_rate=0.01,
        # momentum=0.9,
        # weights_file="weights.pkl",
        # weights_file="weights_87acc_20ep_noreg_softplus.pkl"
        # weights_file=None,
    )

    test = [i for i in range(10)]
    print(function.normalize(test))

    # x_train = function.PCA(x_train, n_components)
    # x_test = function.PCA(x_test, n_components)
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
