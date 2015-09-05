#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division
import numpy as np

from read_data import get_train_data


def cost_function(X, Y, theta):
    m, n = np.shape(X)
    cost = X * theta - Y
    J = 0.5 / m * cost.T * cost
    return J

def feature_standardization(X):
    """ `var` replace `std` also works
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[0, 0] = 1
    X_normalize = (X - X_mean) / X_std
    X_normalize[:, 0] = 1.0
    return X_normalize, X_mean, X_std

def batch_gradient_descent(X, Y, theta, alpha=0.01, iters=1000):
    """ gradient descent to get local minimum of cost function

    :param theta: theta is an array
    :param iters: iters times
    """
    m, n = np.shape(X)
    cost_history = np.zeros((iters, 1))

    for i in xrange(iters):
        theta -= alpha/m * (X.T * (X * theta - Y))
        cost_history[i] = cost_function(X, Y, theta)
        if cost_history[i] < 0.001: break
    return theta, cost_history


def stochastic_gradient_descent(X, Y, theta, alpha=0.01):
    theta -= alpha/m * (X.T * (X * theta - Y))
    return theta

def train(n=806148):
    cnt = 0
    phase = 1000
    cost_history = []
    theta = np.zeros((n-3,1))

    for data in get_train_data():
        # m, n = len(data), len(data[0])
        X, Y = [], []
        for i in data:
            X.append( i[:-3] )
            Y.append( i[-3:] )
        theta = stochastic_gradient_descent(np.mat(X), np.mat(Y), theta, 0.001)

        cnt += 1
        if cnt == phase:
            cnt = 0
            cost = cost_function(np.mat(X), np.mat(Y), theta)
            if cost < 0.00001: break
            cost_history.append(cost)
    return theta, cost_history


if __name__ == '__main__':
    theta, cost_history = train()
    print(cost_history)

    import cPickle
    with open('result.dump', 'w') as fd:
        cPickle.dump((theta, cost_history), fd)

