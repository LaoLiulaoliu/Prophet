#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def pca(X, topN=999):
    X_mean = np.mean(X, axis=0)
    X_anomaly = X - X_mean # 距平
    covariance = np.cov(X_anomaly, rowvar=0) # (n * n)
    eig_value, eig_vector = np.linalg.eig(np.mat(covariance)) # eig_vector (n * n)
    eig_value_idx = np.argsort(eig_value)
    top_eig_vector_idx = eig_value_idx[:-(topN+1):-1]
    top_eig_vector = eig_vector[:, top_eig_vector_idx]
    low_dimension = X_anomaly * top_eig_vector #(m * n) * (n * topN)
    refactor_X = low_dimension * top_eig_vector.T + X_mean
    return low_dimension, refactor_X


def variance_precentage(X):
    def plot_principle(y):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(y)), y, marker='^')
        plt.xlabel('Principal component number')
        plt.ylabel('percentage of variance')
        plt.show()

    X_mean = np.mean(X, axis=0)
    X_anomaly = X - X_mean # 距平
    covariance = np.cov(X_anomaly, rowvar=0) # (n * n)
    print('cov over')
    eig_value, eig_vector = np.linalg.eig(np.mat(covariance)) # eig_vector (n * n)
    print('eigenvalue over')
    sorted_eig_value = sorted(eig_value, reverse=True)
    variance_percentage = np.asarray(sorted_eig_value) / sum(sorted_eig_value) * 100
    plot_principle(variance_percentage)
    return variance_percentage


def replace_nan_with_mean(X):
    X = np.mat(X)
    _, n = np.shape(X)
    for i in range(n):
        mean_val = np.mean( X[np.nonzero( ~np.isnan(X[:, i]) )[0], i] )
        X[np.nonzero( np.isnan(X[:, i]) )[0], i] = mean_val
    return X


if __name__ == '__main__':
    from word_vec import WordVec
    wv = WordVec()
    words, Y = wv.load_words()
    bag_words, _ = wv.bag_word(words)

    variance_precentage(np.mat(bag_words))

