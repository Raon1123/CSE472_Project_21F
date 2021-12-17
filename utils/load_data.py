import pickle
import os
import numpy as np
import urllib.request as rq

def unpickle(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # parse data
    X = dict[b'data']
    y = dict[b'labels']

    return X, y


def get_cifar10(train=True, dataroot='./data'):
    dir_PATH = os.path.join(dataroot, 'cifar-10-python/cifar-10-batches-py')

    if train:
        batch = 5
        for idx in range(1, batch+1):
            file = 'data_batch_' + str(idx)
            file_PATH = os.path.join(dir_PATH, file)
            
            batch_X, batch_y = unpickle(file_PATH)
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)

            if idx == 1:
                X = batch_X
                y = batch_y
            else:
                X = np.concatenate((X, batch_X))
                y = np.concatenate((y, batch_y))
            
    else:
        file = 'test_batch'
        file_PATH = os.path.join(dir_PATH, file)
        X, y = unpickle(file_PATH)
        X = np.array(X)
        y = np.array(y)
    
    # one-hot encoding
    # y = np.eye(10)[y]

    return X, y


def get_data(dataset='cifar10', train=True, dataroot='./data'):
    if dataset == 'cifar10':
        X, y = get_cifar10(train, dataroot)
    else:
        print("Wrong dataset")
        return
    
    return X, y