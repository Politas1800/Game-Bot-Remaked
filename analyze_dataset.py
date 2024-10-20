import numpy as np
from get_dataset import get_dataset

def analyze_dataset():
    X, X_test, Y, Y_test = get_dataset()

    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    print(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    print(f'X min: {X.min()}, X max: {X.max()}')
    print(f'Y unique values: {set(Y.flatten())}')
    print(f'Y_test unique values: {set(Y_test.flatten())}')

    print(f'Number of samples per class in training set:')
    unique, counts = np.unique(Y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f'Class {u}: {c} samples')

if __name__ == '__main__':
    analyze_dataset()
