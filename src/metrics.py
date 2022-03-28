import math

import numpy as np


def rmse_spearman(matrix_predicted, matrix_actual, path):
    """
    Calculates the RMSE error and Spearman correlation

    Parameters:
    matrix_predicted: this matrix contains detected values
    matrix_test: this matrix contains original values
    path: this is the path to the test file containing testing instances
    """

    total = 0.0
    no_instances = 0

    for line in open(path, 'r'):
        values = line.split(',')
        r, c = int(values[0]) - 1, int(values[1]) - 1
        total += math.pow((matrix_actual[r, c] - (matrix_predicted[r, c])), 2)
        no_instances += 1

    rho = 1 - (6 * total) / (no_instances * (math.pow(no_instances, 2) - 1))

    rmse, spear = math.sqrt(total) / no_instances, rho
    print(f'\nRMSE Error: {rmse}')
    print(f'Spearman Correlation: {spear * 100}%')


def precision_on_top_k(matrix_predicted, matrix_actual, k = 100):

    # find top k movies based on their average rating
    # create mask so that avg ratings are calculated over the same instances in both train and test matrix
    matrix_actual = matrix_actual.toarray()
    zero_values = (matrix_actual==0.0)
    matrix_predicted[zero_values] = 0

    # first find according to ratings in train matrix
    movie_mean_predicted = (np.squeeze(np.array(matrix_predicted.sum(axis = 0)/(matrix_predicted!=0).sum(axis = 0))))
    movie_mean_predicted[np.isnan(movie_mean_predicted)] = 0
    movie_mean_predicted_sorted = sorted(movie_mean_predicted.tolist(), reverse = True)[:k]

    # now find according to ratings in testing matrix
    movie_mean_actual = (np.squeeze(np.array(matrix_actual.sum(axis = 0)/(matrix_actual!=0).sum(axis = 0))))
    movie_mean_actual[np.isnan(movie_mean_actual)] = 0
    movie_mean_actual_sorted = sorted(movie_mean_actual.tolist(), reverse = True)[:k]

    # compare both lists to find precision
    fp = 0
    for i in range(k):
        fp += abs(movie_mean_predicted_sorted[i] - movie_mean_actual_sorted[i])
    fp = fp / k

    print(f'Precision on top {k}: {((k - fp) / k) * 100}%\n')
