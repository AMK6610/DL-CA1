import numpy as np
from network import Network
import mnist
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
print("Training...")


# x_train = X_train / 255 #normalization
# x_test = X_test / 255 #normalization

def main():
    x_train, y_train, x_test, y_test = read_data()
    net = Network(
        num_nodes_in_layers=[784, 200, 10],
        batch_size=128,
        num_epochs=50,
        learning_rate=0.001,
    )

    net.train(x_train, y_train)

    print("Testing...")
    net.test(x_test, y_test)


if __name__ == '__main__':
    main()
