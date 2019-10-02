import numpy as np


N = 10

P = np.ndarray(shape=(N, N))

for i in range(N):
    for j in range(N):
        P[i, j] = 0

print(P)

P[1, 1] = 1

print(P)
