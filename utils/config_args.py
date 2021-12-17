import argparse

"""
Argument parsing
"""

def parse_args (parser):
    # Data
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--dataset', type=str, choice=['cifar10'], default='cifar10')

    # Hyperparameter
    parser.add_argument('--C', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--kernel', type=str, choice=['linear', 'gaussian'], default='gaussian')
    

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.num_feature = 10

    return args