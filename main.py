import argparse
import pickle

from train_model import train_step, test_step
from utils.load_data import get_data
from utils.make_dict import train_bow, get_bow
from utils.config_args import parse_args

"""
Main Framework
"""
def run():
    parser = argparse.ArgumentParser(description="Run Tranformer models.")
    args = parse_args(parser)

    trainX, trainy = get_data(dataset=args.dataset, train=True, dataroot=args.dataroot)

    if args.dataset == 'cifar10':
        trainX = trainX.reshape((-1, 32, 32, 3), order='F')

    if args.loadcluster:
        with open("./cluster.dump", "rb") as f:
            cluster = pickle.load(f)
    else:
        cluster = train_bow(trainX, num_dict=args.dictsize, num_select=10000)
        with open("./cluster.dump", "wb") as f:
            pickle.dump(cluster, f)

    trainFeature = get_bow(trainX, cluster, num_dict=args.dictsize)

    dictargs = {'dataset': args.dataset,
                'model': args.model,
                'kernel': args.kernel,
                'validation': args.validation,
                'C': args.C,
                'sigma': args.sigma,
                'batch': args.batch,
                'train': True,
                'cuda': args.cuda
                }

    models, train_acc_list, valid_acc_list = \
            train_step(args, trainFeature, trainy)

    # Test phase
    testX, testy = get_data(dataset=args.dataset, train=False, dataroot=args.dataroot)
    
    if args.dataset == 'cifar10':
        testX = testX.reshape((-1, 32, 32, 3), order='F')

    testFeature = get_bow(testX, cluster, num_dict=args['dict_size'])

    test_acc_list = test_step(args, testFeature, testy, models)

    print("Test average accuracy:", sum(test_acc_list) / len(test_acc_list))


if __name__ == "__main__":
    run()