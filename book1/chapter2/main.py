import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    value = np.sum(x * w) + b
    return 1 if value > 0 else 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    value = np.sum(x * w) + b
    return 1 if value > 0 else 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    value = np.sum(x * w) + b
    return 1 if value > 0 else 0


def XOR(x1, x2):
    a = NAND(x1, x2)
    b = OR(x1, x2)
    return AND(a, b)


print(AND(1, 1))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0, 0))
print("---")
print(NAND(1, 1))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(0, 0))
print("---")
print(OR(1, 1))
print(OR(1, 0))
print(OR(0, 1))
print(OR(0, 0))
print("---")
print(XOR(1, 1))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(0, 0))
