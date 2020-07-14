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


# x = np.arange(-3, 3, 000000.1)
# y1 = step_function(x)
# y2 = sigmoid(x)
# y3 = relu(x)

# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)

# plt.ylim(-0.1, 1.1)
# plt.show()

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print("Z1", Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print("Z2", Z2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
softmaxY = softmax(A3)

print("Y", Y)
print("softmaxY", softmaxY)
