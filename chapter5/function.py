import numpy as np
# import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)  # オーバーフロー対策
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)

        return grad


def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        original_value = x[i]

        x[i] = original_value + h
        fx_1 = f(x)

        x[i] = original_value - h
        fx_2 = f(x)

        grad[i] = (fx_1 - fx_2) / 2 * h

        x[i] = original_value

    return grad
