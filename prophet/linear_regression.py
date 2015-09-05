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
    m, _ = np.shape(X)
    theta -= alpha/m * (X.T * (X * theta - Y))
    return theta

def train(n=806148):
    cnt = 0
    phase = 1000
    alpha = 0.001
    ambit = 0.00001
    costf_hist, costc_hist, costl_hist = [], [], []
    thetaf, thetac, thetal = np.zeros((n-3,1)), np.zeros((n-3,1)), np.zeros((n-3,1))

    for data in get_train_data():
        X, Yf, Yc, Yl = [], [], [], []
        for i in data:
            X.append( i[:-3] )
            Yf.append( i[-3] )
            Yc.append( i[-2] )
            Yl.append( i[-1] )
        thetaf = stochastic_gradient_descent(np.mat(X), np.mat(Yf).T, thetaf, alpha)
        thetac = stochastic_gradient_descent(np.mat(X), np.mat(Yc).T, thetaf, alpha)
        thetal = stochastic_gradient_descent(np.mat(X), np.mat(Yl).T, thetaf, alpha)

        cnt += 1
        if cnt == phase:
            cnt = 0
            costf = cost_function(np.mat(X), np.mat(Yf).T, thetaf)
            costc = cost_function(np.mat(X), np.mat(Yc).T, thetac)
            costl = cost_function(np.mat(X), np.mat(Yl).T, thetal)
            if costf < ambit and costc < ambit and costl < ambit:
                break
            costf_hist.append(costf)
            costc_hist.append(costc)
            costl_hist.append(costl)
    return thetaf, thetac, thetal, costf_hist, costc_hist, costl_hist


if __name__ == '__main__':
    result = train()
    print(result)

    import cPickle
    with open('result.dump', 'w') as fd:
        cPickle.dump(result, fd)

