import time
import numpy as np
import matplotlib.pyplot as plt
import mnist_reader
from utils import squarify
import utils
import gabor
import bayes


def main():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    now = time.time()
    gabor_filters = gabor.generate_gabor_filters()

    x_train_filtered = np.array([
        gabor.apply_filters(squarify(x), gabor_filters)
        for x in X_train
    ])
    x_test_filtered = np.array([
        gabor.apply_filters(squarify(x), gabor_filters)
        for x in X_test
    ])

    edge_value = 100

    best_err = float('Inf')
    best_a = 0
    best_b = 0

    x_train_estimated = np.array([
        bayes.get_estimators(x, edge_value)
        for x in x_train_filtered
    ])
    x_test_estimated = np.array([
        bayes.get_estimators(x, edge_value)
        for x in x_test_filtered
    ])
    a_values = range(1, 21, 2)
    b_values = range(1, 21, 2)
    best_err, best_a, best_b = bayes.model_selection_nb(x_train_estimated, x_test_estimated, y_train, y_test, a_values, b_values)

    print('najlepszy blad:', best_err)
    print('najlepsze a: ', best_a)
    print('najlepsze b: ', best_b)
    print(time.time() - now, 's')


if __name__ == '__main__':
    main()
