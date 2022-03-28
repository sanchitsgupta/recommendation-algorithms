from os import makedirs
from os.path import join


# ---- Base Config ----
data_path = join('data', 'ml-1m', 'ratings.dat')
processed_files_dir = 'processed_data'

all_instances_file_name = 'ratings_all'
train_file_name = 'ratings_train'
test_file_name = 'ratings_test'

test_perc = 0.25        # percentage of testing instances

collaborative_neighbours = 150      # number of neighbours used in collaborative filtering
concepts = 40                       # number of concepts/eigen values to consider while performing SVD decomposition
CUR_no_cols = 4 * concepts          # number of columns and rows to select while performing CUR
# ----------------

txt_dir = join(processed_files_dir, 'txts')
sparse_dir = join(processed_files_dir, 'sparse_matrices')

makedirs(txt_dir, exist_ok=True)
makedirs(sparse_dir, exist_ok=True)
