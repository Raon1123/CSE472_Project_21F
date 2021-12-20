import numpy as np
import tqdm as tqdm
import math

from utils.load_data import get_data
from models.svm import SVM
from models.decisiontree import RandForest, DecisionTree
from sklearn import svm

def get_features(dataset):
    if dataset == 'cifar10':
        features = list(range(10))
    elif dataset == 'MNIST':
        features = list(range(10))
    
    return features

"""
train models
"""
def train_svm(args, models, trainFeature, trainy, validFeature, validy):
    dataset = args['dataset']
    batch_sz = args['batch']

    features = get_features(dataset)

    trainN = trainy.shape[0]
    validN = validy.shape[0]
    featuresN = len(features)

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


def tree_svm(args, models, trainFeature, trainy, validFeature, validy):
    train_acc_list = []
    valid_acc_list = []

    # Train tree model
    models.fit(trainFeature, trainy)

    # Train accuracy
    pred = models.predict(trainFeature)
    boolean = (pred == trainy)
    acc = np.sum(boolean) / trainy.shape[0] * 100
    train_acc_list.append(acc)

    # Validation accuracy
    pred = models.predict(validFeature)
    boolean = (pred == validy)
    acc = np.sum(boolean) / validy.shape[0] * 100
    valid_acc_list.append(acc)

    return models, train_acc_list, valid_acc_list


"""
train models
"""
def train_step(args, trainFeature, trainy):
    dataset = args['dataset']
    validation = args['validation']

    model_type = args['model']

    C = args['C']
    sigma = args['sigma']
    kernel = args['kernel']

    depth = args['depth']
    forest = args['forest']
    bag_size = args['bag_size']

    labels = get_features(dataset)
    
    # Prepare models
    if model_type == 'custom_SVM':
        models = [SVM(kernel=kernel, C=C, sigma=sigma, cuda=args['cuda']) for i in labels]
    elif model_type == 'decisiontree':
        models = DecisionTree(depth=depth)
    elif model_type == 'randomforest':
        models = RandForest(forest=forest, bag_size=bag_size, depth=depth)
    elif model_type == 'sklearn_SVM':
        models = [svm.SVC(kernel='rbf', C=C) for i in labels]

    # Constants
    trainN = trainFeature.shape[0] # Number of data
    validN = int(trainN * validation) # Number of validation data

    # Prepare validation data
    select_idx = np.random.choice(trainN, validN, replace=False)
    validFeature = trainFeature[select_idx]
    validy = trainy[select_idx]

    # Training
    if model_type == 'custom_SVM' or model_type == 'sklearn_SVM':
        models, train_acc_list, valid_acc_list = train_svm(args, models, trainFeature, trainy, validFeature, validy)
    else:
        models, train_acc_list, valid_acc_list = tree_svm(args, models, trainFeature, trainy, validFeature, validy)
            
    return models, train_acc_list, valid_acc_list


"""
Test models

Return
- test_acc_list: test accuracy list
- test_prec_list: test precision list
- test_recall_list: test recall list
- test_f1_list: test f1 list
"""
def test_step(args, testFeature, testy, models):
    dataset = args['dataset']
    model_type = args['model']
    batch_sz = 2000

    features = get_features(dataset)  

    testN = testFeature.shape[0]

    test_acc_list = []
    test_prec_list = []
    test_recall_list = []
    test_f1_list = []

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

    if model_type == 'custom_SVM' or model_type == 'sklearn_SVM':
        tp_cnt = 0
        tn_cnt = 0
        fp_cnt = 0
        fn_cnt = 0

        for model, feature in zip(models, tqdm.tqdm(features)):
            for batch in testlist:
                testX, testy = batch

                gt_true = (testy == feature)
                gt_false = np.logical_not(gt_true)

                testones = gt_true.astype(np.int8) * 2 - 1

                ret = model.predict(testX)

                tp = np.sum(testones[gt_true] == ret[gt_true])
                tp_cnt += tp

                tn = np.sum(testones[gt_false] == ret[gt_false])
                tn_cnt += tn

                # false positive: gt false, ret true
                fp = np.sum(testones[gt_false] != ret[gt_false])
                fp_cnt += fp

                # false negative: gt true, ret false
                fn = np.sum(testones[gt_true] != ret[gt_true])
                fn_cnt += fn

        acc = (tp_cnt + tn_cnt) / (tp_cnt + fp_cnt + fn_cnt + tn_cnt + 1e-5)
        prec = tp_cnt / (tp_cnt + fp_cnt + 1e-5)
        recall = tp_cnt / (tp_cnt + fn_cnt + 1e-5)
        f1 = 2 * (prec * recall) / (prec + recall + 1e-5)

        test_acc_list.append(acc)
        test_prec_list.append(prec)
        test_recall_list.append(recall)
        test_f1_list.append(f1)
    else:
        pred = models.predict(testFeature)
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for feature in features:
            gt_true = (testy == feature)
            gt_false = np.logical_not(gt_true)

            tp += np.sum(testy[gt_true] == pred[gt_true])
            tn += np.sum(testy[gt_false] == pred[gt_false])
            fp += np.sum(testy[gt_false] != pred[gt_false])
            fn += np.sum(testy[gt_true] != pred[gt_true])

        acc = (tp + tn) / (tp + fp + fn + tn + 1e-5)
        prec = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * (prec * recall) / (prec + recall + 1e-5)

        test_acc_list.append(acc)
        test_prec_list.append(prec)
        test_recall_list.append(recall)
        test_f1_list.append(f1)

    return test_acc_list, test_prec_list, test_recall_list, test_f1_list
