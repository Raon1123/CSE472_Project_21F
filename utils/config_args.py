import argparse

"""
Argument parsing
"""

def parse_args (parser):
    # Data
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--dataset', type=str, choice=['cifar10', 'MNIST'], default='cifar10')

    # Model selection
    parser.add_argument('--model', type=str, choice=['custom_SVM', 'SVM', 'DecisionTree', "RandomForest"], default='gaussian')
    parser.add_argument('--kernel', type=str, choice=['linear', 'gaussian'], default='gaussian')

    # Hyperparameter
    parser.add_argument('--dictsize', type=int, default=100)
    parser.add_argument('--C', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.01)
    
    # Test and Validation
    parser.add_argument('--validation', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=1000)

    # ETC
    parser.add_argument('--loadcluster', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=False) 

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.num_feature = 10
    elif args.dataset == 'MNIST':
        args.num_feature = 10

    return args