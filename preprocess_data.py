import os
from os.path import join
import numpy as np
from scipy import sparse
import time

def form_sparse_matrix(file, name, shape, test = False):
	"""
	Creats the sparse matrix from input data and saves it to data_sparse.npz
	
	Parameters:
	file is the path to the user/movie ratings
	name is the name of the saved normalized sparse matrix

	Returns: the created sparse matrix
	"""
	start = time.time()
	users_list = []
	movies_list = []
	data_list = []
	for i, line in enumerate(open(file)):
		values = line.split(',')
		users_list.append(int(values[0]))
		movies_list.append(int(values[1]))
		data_list.append(int(values[2]))

	no_users, no_movies = max(users_list), max(movies_list)
	row = np.array(users_list, dtype = np.float32)
	col = np.array(movies_list, dtype = np.float32)
	data = np.array(data_list, dtype = np.float32)
	row-=1
	col-=1
	if test:
		sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()
	else:
		sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=(no_users, no_movies)).tocsr()
	# save the sparse matrix
	sparse.save_npz(name + '_original.npz', sparse_matrix)
	print('\nSaved ' + name + '_original.npz in ' + '{0:.2f}'.format(time.time() - start) + ' secs.')
	# print(sparse_matrix.todense())
	return sparse_matrix

def normalize(sparse_matrix, name):
	"""
	Normalizes the sparse matrix by subtracting row mean from each non zero value, and saves the normalized matrix to data_sparse_normalized.npz

	Parameters: 
	sparse_matrix is a scipy csr sparse matrix
	name is the name of the saved normalized sparse matrix
	"""
	start = time.time()
	row_mean = sparse_matrix.sum(1)/(sparse_matrix!=0).sum(1)
	dense_matrix = sparse_matrix.todense()
	r,c = np.where(dense_matrix == 0)
	sparse_matrix-=row_mean
	sparse_matrix[r,c] = 0
	sparse_matrix = sparse.csr_matrix(sparse_matrix)
	# print(sparse_matrix.todense())
	sparse.save_npz(name + '_normalized.npz', sparse_matrix)
	print('\nSaved ' + name + '_normalized.npz in ' + '{0:.2f}'.format(time.time() - start) + ' secs.')

def main():
	dataset_dir = join(os.getcwd(), 'Datasets', 'Movie Lens Movie Recommendation Dataset', 'ml-1m')
	train_file = join(dataset_dir, 'ratings_train.txt')
	test_file = join(dataset_dir, 'ratings_test_200.txt')
	# train_file = 'temp.txt'
	test_file = 'temp_test.txt'
	# train_file = 'temp3.txt'

	# save whole user/movie matrix in sparse form (original and normalized) 
	sparse_matrix = form_sparse_matrix(train_file, train_file.split('\\')[-1][:-4], (0,0))
	normalize(sparse_matrix, train_file.split('\\')[-1][:-4])
	# save test user/movie matrix in sparse form (original and normalized)
	sparse_matrix_test = form_sparse_matrix(test_file, test_file.split('\\')[-1][:-4], sparse_matrix.shape, test = True)
	normalize(sparse_matrix_test, test_file.split('\\')[-1][:-4])

if __name__ == '__main__':
	main()