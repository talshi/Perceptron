import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import copy

# constant values
r = 0.1
t = 0.5
upper_bound = 100

iteration = 0
equality_counter = 0


def calc_w(w):
    w += X[i, 0:3] * d[j]
    return w


def calc_c(c):
    c = X[i, 0:3] * W
    return c


def calc_err(z, n):
    return z - n


def calc_d(r, err):
    return r * err


# inputs
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# desired values
Z = np.array([[1], [1], [1], [0]])

# initial values (zeros)
# W = np.zeros([1, 3])
W = [0, 0, 0]
C = np.array([0, 0, 0])
N = np.zeros([4, 1])

err = np.array([Z - N])

print "Matrix X: "
print X
print "Vector Z: "
print Z
print "Matrix W: "
print W

# index of matrix line
i = 0
j = 0

# loop blocked by upper_bound
while iteration < upper_bound:
    if i >= 4:
        i = 0
    if j >= 4:
        j = 0
    C = calc_c(C)
    sumC = np.sum(C)

    # debugging
    print "C =", X[i, 0:4], "*", W, "=", C
    print "sum of mat c : ", sumC

    # decide value of N for each iteration
    if sumC > t:
        N[j] = 1
    else:
        N[j] = 0

    # calc error and d
    err = calc_err(Z, N)
    # d = r * err
    d = calc_d(r, err)

    # debugging
    print "n ="
    print N
    print "error ="
    print err
    print "d ="
    print d

    prevW = copy.copy(W)
    # W += X[i, 0:3] * d[j]
    W = calc_w(W)

    # debugging
    print('-' * 100)
    print "W", iteration
    print W
    print "N", iteration
    print N
    print('-' * 100)

    # stop condition.
    # check if W = previous W 3 times. if yes, break.
    if (prevW == W).all():
        equality_counter += 1
    else:
        equality_counter = 0
    if equality_counter == 3:
        break

    x21 = x22 = 0
    if iteration != 0:
        x21 = (-W[0] - C[1] - 1 - t) / (W[2] - t)
        x22 = (-W[0] - C[1] + 2 - t) / (W[2] - t)
    x2 = [x21, x22]
    print "x2 =", x2

    # paint graph
    fig = plt.figure(figsize=(15, 7))
    plt.plot(X[0:3, 1], X[0:3, 2], 'ro', ms=10)
    plt.plot(X[3, 1], X[3, 2], 'bo', ms=10)
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.title(W)
    plt.plot([-1, 2], x2)

    iteration += 1
    i += 1
    j += 1

    plt.show()
