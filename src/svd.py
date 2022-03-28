import time

import numpy as np
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator


def svd_sparse(sparse_matrix, no_eigen_values):

    def transpose(x):
        return x.T

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    n, m = sparse_matrix.shape
    X_dot = X_matmat = sparse_matrix.dot
    XH_dot = transpose(sparse_matrix).dot

    XH_X = LinearOperator(
        matvec=matvec_XH_X,
        dtype=sparse_matrix.dtype,
        shape=(min(sparse_matrix.shape), min(sparse_matrix.shape))
    )
    eigvals, eigvec = linalg.eigsh(XH_X, k = no_eigen_values)
    eigvals = np.maximum(eigvals.real, 0)

    # in our case all eigen values are going to be greater than zero
    # create sigma diagnol matrix
    slarge = np.sqrt(eigvals)
    s = np.zeros_like(eigvals)
    s[:no_eigen_values] = slarge

    ularge = X_matmat(eigvec)/slarge
    vhlarge = transpose(eigvec)

    return ularge, s, vhlarge


def svd_retain_energy(sparse_matrix, no_eigen_values, energy = 1):
    u, s, vt = svd_sparse(sparse_matrix, no_eigen_values)
    s_squared_sum = np.square(s).sum()		# sum of square of all eigen values (diagnol elements in s)

    for i in range(s.shape[0]):
        if np.square(s[i:]).sum()<(energy*s_squared_sum):
            break
    i -= 1

    return np.delete(u, np.s_[:i], 1), s[i:], np.delete(vt, np.s_[:i], 0)


def svd(sparse_matrix, no_eigen_values, energy = 1):
    """
    Perform SVD Decomposition on the input sparse_matrix
    Pass the copy of the sparse matrix to keep the original matrix unchanged

    Parameters:
    sparse_matrix : input sparse_matrix
    no_eigen_values: number of largest eigen values desired
    energy: retain energy% of largest eigen values

    Returns : The dot product of U S and Vt matrix
    """

    start = time.time()
    print(f'---- SVD with {energy * 100}% energy ----')

    u,s,vt = svd_retain_energy(sparse_matrix, no_eigen_values, energy)
    svd_matrix = np.dot(np.dot(u,np.diag(s)), vt)

    print('SVD took ' + '{0:.2f}'.format(time.time() - start) + ' secs.')
    return svd_matrix
