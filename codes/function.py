import numpy as np


# activation function
def relu(inputs):
    return np.maximum(inputs, 0)


# output probability distribution function
def softmax(inputs):
    exp = np.exp(inputs)
    return exp / np.sum(exp, axis=1, keepdims=True)


# loss
def cross_entropy(inputs, y):
    indices = np.argmax(y, axis=1).astype(int)
    probability = inputs[np.arange(len(inputs)), indices]  # inputs[0, indices]
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss


def svm_loss(W, X, y):
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]), y]  # http://stackoverflow.com/a/23435843/459241
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum.T
    binary /= num_train
    return loss, binary


# L2 regularization
def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss
