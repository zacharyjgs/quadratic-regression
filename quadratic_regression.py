"""
This program allows one to find a linear or quadratic
regression of a data set and evaluate the accuracy of
the model.
"""

# import various packages
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


# initializes data matrix
m = np.array([[0, 1], [1, 0], [2, -1], [3, 0], [4, 1], [5, 0]])

# projects the "model vector" onto the "data space" and calculates the \
# difference vector (outputs coefficients for each variable/power)


def findFit(data, degree):
    AT = np.ones(data.shape[0])
    for d in range(1, degree + 1):
        AT = np.vstack((data[:, 0]**d, AT))
    A = np.transpose(AT)
    ATAinv = np.linalg.inv(np.dot(AT, A))
    b = data[:, 1]
    a = np.dot(np.dot(ATAinv, AT), b)
    return a


# graphs data and model
def graphFit(data, a, extra=0):
    degree = a.size - 1
    plt.plot(data[:, 0], data[:, 1], 'o')
    x = np.linspace(np.amax(data, axis=0)[0] + extra,
                    np.amin(data, axis=0)[0] - extra, 100)
    yp = a[degree]
    for d in range(degree):
        yp += a[d]*(x**(degree - d))
    plt.plot(x, yp)
    plt.show()


# finds standard deviation of data as compared to the model
def findSDEV(data, a):
    degree = a.size - 1
    x = data[:, 0]
    y = data[:, 1]
    yp = a[degree]
    for d in range(degree):
        yp += a[d]*(x**(degree - d))
    d = y - yp
    var = d**2
    sdev = (np.sum(var)/d.shape[0])**0.5
    return sdev

a = findFit(m, 8)
print a
print findSDEV(m, a)
graphFit(m, a)
