import numpy as np
import tqdm

"""
Generate split function for decision tree
generate function is when true w*x+b>thres
for generation
- fea: selected feature
- w, b: linear separable constant
- thres: thresold for linear separable function
"""
def split_linear_func(fea, w, thres=0.5):
    def func(X):
        # feature selection
        x = X[:, fea]
        ret = (x > w * thres)
        return ret
    return func


def leaf_func(leaf):
    def func(X):
        return leaf
    return func


"""
Split data by func
"""
def split_data(X, y, func):
    boolean = func(X)
    neg_boolean = np.logical_not(boolean)
    pos = [X[boolean], y[boolean]]
    neg = [X[neg_boolean], y[neg_boolean]]
    return pos, neg 


"""
Calculate Entropy
"""
def entropy(y):
    tot_cnt = y.shape[0]
    assert tot_cnt != 0

    _, counts = np.unique(y, return_counts=True)

    ret = 0.0

    for cnt in counts:
        if cnt == 0:
            continue
        prob = cnt / tot_cnt
        ret += -1.0 * prob * np.log2(prob)

    return ret


"""
Calculate information gain
E(X) - E(X|pos)
"""
def info_gain(X, y, func):
    pos, _ = split_data(X, y, func)

    if pos[1].shape[0] == 0:
        return 0.0

    prev_ent = entropy(y)
    next_ent = entropy(pos[1])

    return prev_ent - next_ent


class NODE():
    def __init__(self, func):
        self.left = None
        self.right = None
        self.func = func

    def push_left(self, left):
        self.left = left

    def push_right(self, right):
        self.right = right

    def get_func(self):
        return self.func

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class DecisionTree():
    def __init__(self, depth=20):
        self.depth = depth
        self.root = None

    def fit(self, X, y):
        trainN, featureN = X.shape

        samples = [X, y]
        self.root = self.train_node(samples ,self.depth)

        return self

    def train_node(self, samples, depth):
        X, y = samples
        trainN, featureN = X.shape

        bin_cnt = np.bincount(y)
        if (depth == 0) or (np.max(bin_cnt) == trainN):
            max_freq = np.argmax(bin_cnt)
            func = leaf_func(max_freq)
            return NODE(func)

        func = self.function_select(X, y)
        left_data, right_data = split_data(X, y, func)

        ret = NODE(func)
        ret.push_left(self.train_node(left_data, depth-1))
        if right_data[1].shape[0] != 0:
            ret.push_right(self.train_node(right_data, depth-1))
        else:
            max_freq = np.argmax(bin_cnt)
            func = leaf_func(max_freq)
            ret.push_right(NODE(func))

        return ret

    def function_select(self, X, y):
        trainN, featureN = X.shape

        max_info = 0.0
        best_feature = None
        best_w = None
        best_thres = None

        # Stupid algorithm
        while best_feature == None:
            for feature in range(featureN):
                w = np.random.choice([-1,1])
                thres = w * np.random.randint(0, 60)
                func = split_linear_func(feature, w, thres)
                info = info_gain(X, y, func)
                if info > max_info and (info is not np.inf):
                    max_info = info
                    best_feature = feature
                    best_w = w
                    best_thres = thres
        
        return split_linear_func(best_feature, best_w, best_thres)

    def predict(self, X):
        assert self.root is not None

        result = self.branch(X, self.root)

        return result

    def branch(self, X, node):
        func = node.get_func()
        if node.is_leaf():
            return func(X)

        testN = X.shape[0]
        ret = np.zeros(testN)

        boolean = func(X)
        neg_boolean = np.logical_not(boolean)

        ret[boolean] = self.branch(X[boolean], node.get_left())
        ret[neg_boolean] = self.branch(X[neg_boolean], node.get_right())

        return ret


class RandForest():
<<<<<<< HEAD
    def __init__(self, forest=100, bag_size=1000):
        self.forest = [DecisionTree() for i in range(forest)]
=======
    def __init__(self, forest=100, bag_size=1000, depth=100):
        self.forest = [DecisionTree(depth=depth) for i in range(forest)]
>>>>>>> parent of 4bfcc8f (Revert "update model")
        self.bag_size = bag_size

    def fit(self, X, y):
        trainN, featureN = X.shape
        self.featureN = np.unique(y).shape[0]

        for tree in tqdm.tqdm(self.forest):
            bag = np.random.randint(0, trainN, size=self.bag_size)
            selection = np.random.choice(bag, size=trainN, replace=True)
            tree.fit(X[selection,:], y[selection])

        return self

    def predict(self, X):
        testN, _ = X.shape

<<<<<<< HEAD
<<<<<<< HEAD
        prediction = np.zeros(testN, self.featureN)
=======
        prediction = np.zeros((testN, self.forestN))
        pred = np.zeros((testN))
>>>>>>> parent of 4bfcc8f (Revert "update model")
=======
        prediction = np.zeros((testN, self.featureN))
>>>>>>> parent of 46d8f32 (update model)

        for idx, tree in enumerate(tqdm.tqdm(self.forest)):
            prediction[:,idx] = tree.predict(X)
        
        # Voting
        pred = np.argmax(prediction, axis=1)

        return pred 