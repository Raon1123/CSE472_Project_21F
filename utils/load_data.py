import pickle
import os
import numpy as np
import urllib.request as rq
import struct

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

    return X, y


def get_mnist(train=True, dataroot='./data'):
    # Reference
    # https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
    dir_PATH = os.path.join(dataroot, 'mnist')

    if train:
        # Read image
        file_PATH = os.path.join(dir_PATH, "train-images-idx3-ubyte")
        with open(file_PATH, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            X = data.reshape((size, nrows, ncols))

        # Read labels
        file_PATH = os.path.join(dir_PATH, "train-labels-idx1-ubyte")
        with open(file_PATH, 'rb') as f:
            magic, items = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            y = data.reshape((items,))
    else:
        # Read image
        file_PATH = os.path.join(dir_PATH, "t10k-images-idx3-ubyte")
        with open(file_PATH, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            X = data.reshape((size, nrows, ncols))

        # Read labels
        file_PATH = os.path.join(dir_PATH, "t10k-labels-idx1-ubyte")
        with open(file_PATH, 'rb') as f:
            magic, items = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            y = data.reshape((items,))

    return X, y


def get_data(dataset='cifar10', train=True, dataroot='./data'):
    if dataset == 'cifar10':
        X, y = get_cifar10(train, dataroot)
    elif dataset == 'MNIST':
        X, y = get_mnist(train, dataroot)
    else:
        print("Wrong dataset")
        return
    
    return X, y