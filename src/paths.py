from config import *
from scipy import sparse


def get_txt_path(name):
    return join(txt_dir, f'{name}.txt')


def get_txt_path_by_type(type):
    if type == 'train':
        return get_txt_path(train_file_name)
    elif type == 'test':
        return get_txt_path(test_file_name)
    elif type == 'all':
        return get_txt_path(all_instances_file_name)
    else:
        raise ValueError('Invalid value for type')


def get_sparse_path(name, normalized = False):
    suffix = '_normalized' if normalized else '_original'
    return join(sparse_dir, f'{name}{suffix}.npz')


def get_sparse_path_by_type(type, normalized = False):
    if type == 'train':
        return get_sparse_path(train_file_name, normalized)
    elif type == 'test':
        return get_sparse_path(test_file_name, normalized)
    elif type == 'all':
        return get_sparse_path(all_instances_file_name, normalized)
    else:
        raise ValueError('Invalid value for type')


def load_sparse_matrix(type, normalized = False):
    filepath = get_sparse_path_by_type(type, normalized)
    return sparse.load_npz(filepath)
