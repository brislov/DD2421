import random

from scipy.optimize import minimize
import numpy as np


def K(x, y,):
    """ Linear kernal function. """
    return np.dot(x, y)


def objective(alpha):
    """ Takes a vector alpha and returns a scalar value. Implementation of (4). """
    double_sum = 0
    for i in range(N):
        for j in range(N):
            double_sum += alpha[i] * alpha[j] * P[i, j]

    return 0.5 * double_sum - np.sum(alpha)


def zerofun(alpha):
    """ Constraint for minimize(). Implementation of the second constraint from (10). """
    return np.sum([alpha[i] * t[i] for i in range(N)])


if __name__ == '__main__':

    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]  # Number of rows (samples)

    permute = list(range(N))
    random.shuffle(permute)
    x = inputs = inputs[permute, :]
    t = targets = targets[permute]

    start = np.zeros(N)  # Initialise alpha vector
    C = None  # Upper bound for alpha. For no upper bound, C = None.
    B = [(0, C) for n in range(N)]  # Bounds for each alpha element, 0 <= alpha <= C.
    XC = {'type': 'eq', 'fun': zerofun}

    # To improve efficiency some of the calculations are done in advance and stored in matrix P.
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = t[i] * t[j] * K(x[i], x[j])

    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']

    alpha = np.array([round(a, 5) for a in alpha])  # Zero out the almost alphas which are almost zero.

    print(alpha)
