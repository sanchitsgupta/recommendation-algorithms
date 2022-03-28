import numpy as np

from config import collaborative_neighbours, concepts, CUR_no_cols
from paths import load_sparse_matrix, get_txt_path_by_type
from collaborative import collaborative_filtering
from svd import svd
from cur import cur
from metrics import rmse_spearman, precision_on_top_k


np.set_printoptions(precision = 5)
np.seterr(divide='ignore', invalid='ignore')


def main():

    # load all matrices
    all_orig = load_sparse_matrix('all')
    all_norm = load_sparse_matrix('all', normalized=True)
    train_orig = load_sparse_matrix('train')
    train_norm = load_sparse_matrix('train', normalized=True)
    test_orig = load_sparse_matrix('test')
    test_norm = load_sparse_matrix('test', normalized=True)
    test_txt_path = get_txt_path_by_type('test')

    # perform collaborative filtering with and without baseline approach
    collab_matrix = collaborative_filtering(train_norm, train_orig, test_orig, collaborative_neighbours)
    rmse_spearman(collab_matrix, test_orig, test_txt_path)
    precision_on_top_k(collab_matrix, all_orig)

    collab_matrix_baseline = collaborative_filtering(
    	train_norm, train_orig, test_orig, collaborative_neighbours, baseline=True
    )
    rmse_spearman(collab_matrix_baseline, all_orig, test_txt_path)
    precision_on_top_k(collab_matrix_baseline, all_orig)

    # perform svd
    for energy in [1, 0.9]:
        svd_matrix = svd(train_norm, concepts, energy)
        rmse_spearman(svd_matrix, test_norm, test_txt_path)
        precision_on_top_k(svd_matrix, all_norm)

    # perform cur
    for energy in [1, 0.9]:
        cur_matrix = cur(train_norm, CUR_no_cols, concepts, energy)
        rmse_spearman(cur_matrix, test_norm, test_txt_path)
        precision_on_top_k(cur_matrix, all_norm)


if __name__ == '__main__':
    main()
