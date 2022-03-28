import time
from operator import itemgetter
import math

import numpy as np
from scipy import sparse

from svd import svd_retain_energy


def intersection_sparse(sparse_matrix, rows_list_original, cols_list_original):
    """
    Parameters:
    sparse_matrix: the original matrix in sparse form
    rows_prob: a list of the indices of the selected rows for R matrix
    col_list: a list of the indices of the selected columns for C matrix

    Returns:
    sparse matrix W such that W consists of elements sparse_matrix[i,j] for all i, j in rows_list, cols_list respectively
    """

    rows_list, cols_list = [], []			# modified row and column list to create intersection matrix
    no_cols = len(rows_list_original)

    for item in rows_list_original:
        rows_list.extend([item] * no_cols)

    cols_list.extend(cols_list_original * no_cols)
    data_list = [sparse_matrix[r,cols_list[i]] for i, r in enumerate(rows_list)]

    rows_list, cols_list, zero_indices = [], [], []
    for i in range(no_cols):
        rows_list.extend([i] * no_cols)
    cols_list.extend(list(range(no_cols)) * no_cols)

    # delete zero values from data_list and corresponding rows and cols from rows_list and cols_list
    for i, item in enumerate(data_list):
        if item == 0:
            zero_indices.append(i)
    for index in sorted(zero_indices, reverse = True):
        del data_list[index]
        del rows_list[index]
        del cols_list[index]

    row = np.array(rows_list, dtype = np.float32)
    col = np.array(cols_list, dtype = np.float32)
    data = np.array(data_list, dtype = np.float32)

    # form sparse intersection matrix
    W = sparse.coo_matrix((data, (row, col)), shape=(no_cols, no_cols)).tocsr()

    return W


def cur(sparse_matrix, no_cols, no_eigen_values, energy = 1):
    """
    Perform CUR Decomposition on the input sparse_matrix

    Parameters:
    sparse_matrix : input sparse_matrix
    no_cols: number of columns and rows to select
    no_eigen_values: number of largest eigen values desired while performing SVD on W matrix
    energy: retain energy% of largest eigen values

    Returns : The dot product of C U and R matrix
    """

    start = time.time()
    print(f'---- CUR with {energy * 100}% energy ----')

    def select_columns(sparse_matrix_csc, select_col = True):

        sparse_copy = sparse_matrix_csc.copy()
        sparse_matrix_csc = sparse_matrix_csc.power(2)
        total_sum = sparse_matrix_csc.sum()

        col_prob = []	# col_prob contains (indices of column, probabilty of that column)
        for c in range(sparse_matrix_csc.shape[1]):
            col_prob.append((c,sparse_matrix_csc.getcol(c).sum() / total_sum))

        # discard columns with zero frobenius norm
        zero_indices = []
        for i, item in enumerate(col_prob):
            if item[1] == 0:
                zero_indices.append(i)
        for index in sorted(zero_indices, reverse = True):
            del col_prob[index]

        # randomly sample no_cols from the matrix
        # col_prob = random.sample(col_prob, no_cols)
        col_prob.sort(key = itemgetter(1), reverse = True)
        del col_prob[no_cols:]
        col_prob.sort(key = itemgetter(0))
        C = sparse.lil_matrix((sparse_copy.shape[0], no_cols))

        for i in range(no_cols):
            C[:,i] = sparse_copy.getcol(col_prob[i][0])/math.sqrt(no_cols*col_prob[i][1])
            # C[:,i] = sparse_copy[:, col_prob[i][0]]/math.sqrt(no_cols*col_prob[i][1])
        if select_col:
            return C.tocsc(), col_prob
        else:
            return C.transpose().tocsc(), col_prob

    # print(sparse_matrix.todense())

    # select columns to fill C matrix
    C, col_prob = select_columns(sparse_matrix.tocsc())

    # select rows to fill R matrix
    R, row_prob = select_columns(sparse_matrix.transpose().tocsc(), select_col=False)

    # create W matrix (intersection of C and R)
    W = intersection_sparse(sparse_matrix, sorted([x[0] for x in row_prob]), sorted([x[0] for x in col_prob]))
    print('Building C, R, W matrix took ' + '{0:.2f}'.format(time.time() - start) + ' secs.')

    # perform svd on W
    # x,z,yt = linalg.svds(W, k = no_eigen_values)
    x, z, yt = svd_retain_energy(W, no_eigen_values, energy)

    # form U matrix
    U = np.dot(np.dot(np.transpose(yt), np.linalg.matrix_power(np.diag(np.reciprocal(z)),2)), np.transpose(x))
    # U = np.dot(np.dot(np.transpose(yt), np.linalg.matrix_power(np.reciprocal(z),2)), np.transpose(x))

    cur_matrix = np.dot(np.dot(C.todense(), U), R.todense())
    print('CUR Decomposition took ' + '{0:.2f}'.format(time.time() - start) + ' secs.')

    return cur_matrix
