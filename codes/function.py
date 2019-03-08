import numpy as np


def relu(inputs):
    return np.maximum(inputs, 0)


def d_relu(inputs, delta):
    delta[inputs <= 0] = 0
    return delta


def sigmoid(inputs):
    d = np.clip(inputs, -500, 500)
    d = 1.0 / (1 + np.exp(-d))
    return d
    # return 1.0 / (1.0 + np.exp(-x))


def softmax(inputs):
    exp = np.exp(inputs)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softplus(x):
    return np.log(1 + np.exp(x))


def d_softplus(x, delta):
    return np.multiply(delta, sigmoid(x))


def lrelu(x, alpha=0.2):
    return np.where(x > 0, x, x * alpha)


def d_lrelu(inputs, delta, alpha=0.2):
    dx = np.ones_like(inputs)
    dx[inputs < 0] = alpha
    return np.multiply(dx, delta)


def svm_loss(W, X, y):
    loss = 0.0
    num_train = X.shape[0]
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]), y]
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum.T
    binary /= num_train
    return loss, binary


def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss


def PCA(x, n_components):
    M = np.mean(x.T, axis=1)
    C = x - M
    V = np.cov(x.T)
    values, vectors = np.linalg.eig(V)
    P = vectors.T.dot(C.T)
    return np.array(P.T[:, 0:n_components])


def normalize(x):
    res = []
    for item in x:

        res.append((item - np.mean(item)) / np.sqrt(np.var(item)))
    return np.array(res)

