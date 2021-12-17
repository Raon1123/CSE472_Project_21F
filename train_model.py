import numpy as np
import tqdm as tqdm
import math

from utils.load_data import get_data
from models.svm import SVM
from sklearn import svm

"""
train models
"""
def train_step(args, trainFeature, trainy):
    dataset = args['dataset']
    dataroot = args['dataroot']
    validation = args['validation']
    batch_sz = args['batch']

    model_type = args['model']

    C = args['C']
    sigma = args['sigma']
    kernel = args['kernel']

    if dataset == 'cifar10':
        features = list(range(10))
    
    # Prepare models
    if model_type == 'custom_SVM':
        models = [SVM(kernel=kernel, C=C, sigma=sigma) for i in features]
    else:
        models = [svm.SVC(kernel='rbf', C=C) for i in features]

    # Constants
    trainN = trainFeature.shape[0] # Number of data
    validN = int(trainN * validation) # Number of validation data
    featuresN = len(features) # Number of feature

    # Prepare validation data
    select_idx = np.random.choice(trainN, validN, replace=False)
    validFeature = trainFeature[select_idx]
    validy = trainy[select_idx]

    # Store accuracy
    train_acc_list = []
    valid_acc_list = []

    # one versus whole
    for idx in tqdm.tqdm(range(featuresN)):
        model = models[idx]
        feature = features[idx]
        
        # Select positive samples
        select_idx = (trainy == feature)
        posN = np.sum(select_idx)
        posX = trainFeature[select_idx]
        
        # Select negative samples
        neg_select_idx = np.random.choice(trainN - posN, posN, replace=False)
        negX = trainFeature[np.logical_not(select_idx)][neg_select_idx]

        # Concatenate samples
        X = np.vstack([posX, negX])
        y = np.hstack([np.ones(posN), -1.0*np.ones(posN)])

        # shuffle
        shuffle_idx = np.arange(2*posN)
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Divide batch
        batchN = math.ceil(2*posN/batch_sz)
        batch = []

        for i in range(batchN):
            if i != batchN - 1:
                batchX = X[i*batch_sz:(i+1)*batch_sz]
                batchy = y[i*batch_sz:(i+1)*batch_sz]
            else:
                batchX = X[i*batch_sz:]
                batchy = y[i*batch_sz:]
            batch.append([batchX, batchy])

        # For counting
        corr_cnt = 0 # Correction counting
        tot_cnt = 0 # Total counting

        for batch_idx in range(batchN):
            batchX, batchy = batch[batch_idx]

            _ = model.fit(batchX, batchy)
            train_ret = model.predict(batchX)
            correction = (train_ret == batchy)

            tot_cnt += batchy.shape[0]
            corr_cnt += np.sum(correction)

        train_acc_list.append(100.0 * corr_cnt / tot_cnt)

        # Validation
        valid_ret = model.predict(validFeature)

        valid_ones = (validy == feature).astype(np.double) * 2 - 1
        correction = (valid_ret == valid_ones)
        valid_cnt = np.sum(correction)

        #valid_correction[:, idx] = correction
        valid_acc_list.append(100.0 * valid_cnt / validN)
            
    return models, train_acc_list, valid_acc_list


def test_step(args, testFeature, testy, models):
    dataset = args['dataset']
    dataroot = args['dataroot']
    dict_size = args['dict_size']
    batch_sz = 2000

    if dataset == 'cifar10':
        features = list(range(10))

    testN = testFeature.shape[0]

    test_corrs = np.zeros((testN, len(features)))
    test_acc_list = []

    batchN = math.ceil(testN/batch_sz)
    testlist = []

    for i in range(batchN):
        if i != batchN - 1:
            batchX = testFeature[i*batch_sz:(i+1)*batch_sz]
            batchy = testy[i*batch_sz:(i+1)*batch_sz]
        else:
            batchX = testFeature[(batchN-1)*batch_sz:]
            batchy = testy[(batchN-1)*batch_sz:]

        testlist.append([batchX, batchy])

    for model, feature in zip(models, tqdm.tqdm(features)):
        corr_cnt = 0
        tot_cnt = 0

        for batch in testlist:
            testX, testy = batch
            testones = (testy == feature).astype(np.int8) * 2 - 1

            ret = model.predict(testX)

            corr_cnt += np.sum(ret == testones)
            tot_cnt += testy.shape[0]

        test_acc_list.append(100.0 * corr_cnt / tot_cnt)

    return test_acc_list