import numpy as np
from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator

def _augmented_orthonormal_cols(x, k):
	# extract the shape of the x array
	n, m = x.shape
	# create the expanded array and copy x into it
	y = np.empty((n, m+k), dtype=x.dtype)
	y[:, :m] = x
	# do some modified gram schmidt to add k random orthonormal vectors
	for i in range(k):
		# sample a random initial vector
		v = np.random.randn(n)
		if np.iscomplexobj(x):
			v = v + 1j*np.random.randn(n)
		# subtract projections onto the existing unit length vectors
		for j in range(m+i):
			u = y[:, j]
			v -= (np.dot(v, u.conj()) / np.dot(u, u.conj())) * u
		# normalize v
		v /= np.sqrt(np.dot(v, v.conj()))
		# add v into the output array
		y[:, m+i] = v
	# return the expanded array
	return y


def _augmented_orthonormal_rows(x, k):
	return _augmented_orthonormal_cols(x.T, k).T


def _herm(x):
	return x.T.conj()

def svds(sparse_matrix, k=6, ncv=None, tol=0, which='LM', v0=None,
		 maxiter=None, return_singular_vectors=True):
	"""Compute the largest k singular values/vectors for a sparse matrix.

	Parameters
	----------
	sparse_matrix : {sparse matrix, LinearOperator}
		sparse_matrixrray to compute the SVD on, of shape (M, N)
	k : int, optional
		Number of singular values and vectors to compute.
		Must be 1 <= k < min(sparse_matrix.shape).
	ncv : int, optional
		The number of Lanczos vectors generated
		ncv must be greater than k+1 and smaller than n;
		it is recommended that ncv > 2*k
		Default: ``min(n, max(2*k + 1, 20))``
	tol : float, optional
		Tolerance for singular values. Zero (default) means machine precision.
	which : str, ['LM' | 'SM'], optional
		Which `k` singular values to find:

			- 'LM' : largest singular values
			- 'SM' : smallest singular values

		.. versionadded:: 0.12.0
	v0 : ndarray, optional
		Starting vector for iteration, of length min(sparse_matrix.shape). Should be an
		(approximate) left singular vector if N > M and a right singular
		vector otherwise.
		Default: random

		.. versionadded:: 0.12.0
	maxiter : int, optional
		Maximum number of iterations.

		.. versionadded:: 0.12.0
	return_singular_vectors : bool or str, optional
		- True: return singular vectors (True) in addition to singular values.

		.. versionadded:: 0.12.0

		- "u": only return the u matrix, without computing vh (if N > M).
		- "vh": only return the vh matrix, without computing u (if N <= M).

		.. versionadded:: 0.16.0

	Returns
	-------
	u : ndarray, shape=(M, k)
		Unitary matrix having left singular vectors as columns.
		If `return_singular_vectors` is "vh", this variable is not computed,
		and None is returned instead.
	s : ndarray, shape=(k,)
		The singular values.
	vt : ndarray, shape=(k, N)
		Unitary matrix having right singular vectors as rows.
		If `return_singular_vectors` is "u", this variable is not computed,
		and None is returned instead.


	Notes
	-----
	This is a naive implementation using sparse_matrixRPsparse_matrixCK as an eigensolver
	on sparse_matrix.H * sparse_matrix or sparse_matrix * sparse_matrix.H, depending on which one is more efficient.

	"""
	if not (isinstance(sparse_matrix, LinearOperator) or isspmatrix(sparse_matrix)):
		sparse_matrix = np.asarray(sparse_matrix)

	n, m = sparse_matrix.shape

	if k <= 0 or k >= min(n, m):
		raise ValueError("k must be between 1 and min(sparse_matrix.shape), k=%d" % k)

	if isinstance(sparse_matrix, LinearOperator):
		if n > m:
			X_dot = sparse_matrix.matvec
			X_matmat = sparse_matrix.matmat
			XH_dot = sparse_matrix.rmatvec
		else:
			X_dot = sparse_matrix.rmatvec
			XH_dot = sparse_matrix.matvec

			dtype = getattr(sparse_matrix, 'dtype', None)
			if dtype is None:
				dtype = sparse_matrix.dot(np.zeros([m,1])).dtype

			# sparse_matrix^H * V; works around lack of LinearOperator.adjoint.
			# XXX This can be slow!
			def X_matmat(V):
				out = np.empty((V.shape[1], m), dtype=dtype)
				for i, col in enumerate(V.T):
					out[i, :] = sparse_matrix.rmatvec(col.reshape(-1, 1)).T
				return out.T

	else:
		if n > m:
			X_dot = X_matmat = sparse_matrix.dot
			XH_dot = _herm(sparse_matrix).dot
		else:
			XH_dot = sparse_matrix.dot
			X_dot = X_matmat = _herm(sparse_matrix).dot

	def matvec_XH_X(x):
		return XH_dot(X_dot(x))

	XH_X = LinearOperator(matvec=matvec_XH_X, dtype=sparse_matrix.dtype,
						  shape=(min(sparse_matrix.shape), min(sparse_matrix.shape)))

	# Get a low rank approximation of the implicitly defined gramian matrix.
	# This is not a stable way to approach the problem.
	eigvals, eigvec = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter,
								  ncv=ncv, which=which, v0=v0)

	# In 'LM' mode try to be clever about small eigenvalues.
	# Otherwise in 'SM' mode do not try to be clever.
	if which == 'LM':

		# Gramian matrices have real non-negative eigenvalues.
		eigvals = np.maximum(eigvals.real, 0)

		# Use the sophisticated detection of small eigenvalues from pinvh.
		t = eigvec.dtype.char.lower()
		factor = {'f': 1E3, 'd': 1E6}
		cond = factor[t] * np.finfo(t).eps
		cutoff = cond * np.max(eigvals)

		# Get a mask indicating which eigenpairs are not degenerately tiny,
		# and create the re-ordered array of thresholded singular values.
		above_cutoff = (eigvals > cutoff)
		nlarge = above_cutoff.sum()
		nsmall = k - nlarge
		slarge = np.sqrt(eigvals[above_cutoff])
		s = np.zeros_like(eigvals)
		s[:nlarge] = slarge
		if not return_singular_vectors:
			return s

		if n > m:
			vlarge = eigvec[:, above_cutoff]
			ularge = X_matmat(vlarge) / slarge if return_singular_vectors != 'vh' else None
			vhlarge = _herm(vlarge)
		else:
			ularge = eigvec[:, above_cutoff]
			vhlarge = _herm(X_matmat(ularge) / slarge) if return_singular_vectors != 'u' else None

		u = _augmented_orthonormal_cols(ularge, nsmall) if ularge is not None else None
		vh = _augmented_orthonormal_rows(vhlarge, nsmall) if vhlarge is not None else None

	elif which == 'SM':

		s = np.sqrt(eigvals)
		if not return_singular_vectors:
			return s

		if n > m:
			v = eigvec
			u = X_matmat(v) / s if return_singular_vectors != 'vh' else None
			vh = _herm(v)
		else:
			u = eigvec
			vh = _herm(X_matmat(u) / s) if return_singular_vectors != 'u' else None

	else:

		raise ValueError("which must be either 'LM' or 'SM'.")

	return u, s, vh
