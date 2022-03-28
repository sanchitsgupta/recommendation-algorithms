import random

import numpy as np
from scipy import sparse

from config import data_path, test_perc
from paths import get_txt_path_by_type, get_sparse_path_by_type


np.seterr(divide='ignore', invalid='ignore')


def split_train_test():

	# read main file
	ratings = []
	no_users, no_movies = -float('inf'), -float('inf')

	for line in open(data_path, 'r'):
		words = line.split('::')
		ratings.append((words[0], words[1], words[2]))
		no_users = max(no_users, int(words[0]))
		no_movies = max(no_movies, int(words[1]))

	print(f'Number of ratings: {len(ratings)}')
	print(f'Number of users: {no_users}, Number of movies: {no_movies}')

	# create train and test files
	random.shuffle(ratings)
	no_test = int(test_perc * (len(ratings) + 1))

	all_path = get_txt_path_by_type('all')
	train_path = get_txt_path_by_type('train')
	test_path = get_txt_path_by_type('test')

	with open(train_path, 'a') as train_file, open(test_path, 'a') as test_file, open(all_path, 'a') as all_file:
		for i, item in enumerate(ratings):
			line = item[0] + ',' + item[1] + ',' + item[2] + '\n'

			all_file.write(line)
			if i <= no_test:
				test_file.write(line)
			else:
				train_file.write(line)

	return no_users, no_movies


def form_sparse_matrix(type, shape):
	"""
	Creats the sparse matrix from input data and saves it to data_sparse.npz

	Parameters:
	type is the type of file to be processed. Allowed values are 'train', 'test' and 'all'.
	shape should be (no_users, no_movies)

	Returns: the created sparse matrix
	"""

	filepath = get_txt_path_by_type(type)

	users_list, movies_list, data_list = [], [], []
	for line in open(filepath):
		values = line.split(',')
		users_list.append(int(values[0]))
		movies_list.append(int(values[1]))
		data_list.append(int(values[2]))

	row = np.array(users_list, dtype = np.float32)
	col = np.array(movies_list, dtype = np.float32)
	row, col = row - 1, col - 1

	data = np.array(data_list, dtype = np.float32)
	sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()

	# save the sparse matrix
	sparse_filepath = get_sparse_path_by_type(type)
	sparse.save_npz(sparse_filepath, sparse_matrix)
	print(f'Saved {sparse_filepath}')

	return sparse_matrix


def normalize(sparse_matrix, type):
	"""
	Normalizes the sparse matrix by subtracting row mean from each non zero value,
	and saves the normalized matrix to data_sparse_normalized.npz

	Parameters:
	sparse_matrix is a scipy csr sparse matrix
	type is the type of file to be processed. Allowed values are 'train', 'test' and 'all'.
	"""

	row_mean = sparse_matrix.sum(1)/(sparse_matrix!=0).sum(1)
	dense_matrix = sparse_matrix.todense()
	r,c = np.where(dense_matrix == 0)
	sparse_matrix -= row_mean
	sparse_matrix[r,c] = 0
	sparse_matrix = sparse.csr_matrix(sparse_matrix)

	# save the normalized matrix
	sparse_filepath = get_sparse_path_by_type(type, normalized=True)
	sparse.save_npz(sparse_filepath, sparse_matrix)
	print(f'Saved {sparse_filepath}')


def main():

	# split train-test
	print('Splitting into train test ...')
	shape = split_train_test()

	# create and save sparse matrices
	print('\nCreating sparse matrices ...')

	# save all sparse matrices
	for type in ['all', 'train', 'test']:
		sparse_matrix = form_sparse_matrix(type, shape)
		normalize(sparse_matrix, type)


if __name__ == '__main__':
	main()
