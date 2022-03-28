# Recommendation Algorithms

![PyPI pyversions](https://img.shields.io/github/pipenv/locked/python-version/sanchitsgupta/recommendation-algorithms)
![Linux](https://svgshare.com/i/Zhy.svg)
![PyPI license](https://img.shields.io/github/license/sanchitsgupta/recommendation-algorithms)

Implementation of some algorithms used in [Recommender Systems](https://www.wikiwand.com/en/Recommender_system). I implemented three algorithms: Collaborative Filtering, SVD, and CUR Decomposition.

## Running

1. Make sure [Python 3.10+](https://www.python.org/downloads/) is installed.
2. Install [pipenv](https://github.com/kennethreitz/pipenv).
    ```shell
    $ pip install pipenv
    ```
3. Install requirements
    ```shell
    $ pipenv install
    ```
4. Split and process the dataset
    ```shell
    $ pipenv run python src/preprocess.py
    ```
5. Run the algorithms
    ```
    $ pipenv run python src/recommend.py
    ```

## Data

[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) has been used. Data files are present in [data directory](./data/). Data has around 6000 users, 3000 movies and 1 million ratings.

> NOTE: I have not tested the code on other datasets, but with minor changes it should work fine. Though, we might face memory issues on very huge datasets.

## Results

| Algorithm     | RMSE  | Precision on top 100 (%) | Spearman Rank Correlation (%) |
| ------------- |:-------------:|:--------:|:-----:|
| Collaborative Filtering | 0.005736 | 99.24713 | 99.99999 |
| Collaborative Filtering with Baseline Approach | 0.005937 | 99.12928 | 99.99999 |
| SVD | 0.002870 | **99.04599** | 99.99999 |
| SVD with 90% energy | **0.002867** | 99.01968 | 99.99999 |
| CUR | 0.002943 | 98.47204 | 99.99999 |
| CUR with 90% energy | 0.002944 | 98.46733 | 99.99999 |

## Parameters

Test Size: 25%

Collaborative Filtering Neighbourhood Size: 150

SVD and CUR Concepts: 40

CUR Columns/Rows Selected: 160

> NOTE: You can change these parameters in [config.py](./src/config.py)

## Additional Notes

I did this project to get a better understanding of the said algorithms. In a production system, we should use more efficient implementations, such as those available in scipy or scikit-learn.
