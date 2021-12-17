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
def entropy(boolean):
    assert boolean.dtype == np.bool_

    tot_cnt = boolean.shape[0]

    assert tot_cnt != 0

    pos_cnt = np.sum(boolean)
    neg_cnt = tot_cnt - pos_cnt

    prob = pos_cnt / tot_cnt
    pos_entropy = -1.0 * prob * np.log2(prob)
    prob = neg_cnt / tot_cnt
    neg_entropy = -1.0 * prob * np.log2(prob)

    ret = pos_entropy + neg_entropy

    return ret


"""
Calculate information gain
E(X) - E(X|pos)
"""
def info_gain(X, y, func, label):
    pos, _ = split_data(X, y, func)

    tot_cnt = y.shape[0]
    pos_boolean = (pos[1] == label)
    pos_cnt = np.sum(pos_boolean)

    if pos_cnt == 0:
        return 0.0
    if pos_cnt == tot_cnt:
        return 1.0

    pos_prob = pos_cnt / tot_cnt

    prev_ent = entropy((y == label))
    next_ent = entropy(pos_boolean)

    return prev_ent - pos_prob * next_ent


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

        # Select split label
        choice_label = np.random.choice(y)
        
        func = self.function_select(X, y, choice_label)
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

    def function_select(self, X, y, label):
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
                info = info_gain(X, y, func, label)
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
    def __init__(self, forest=100, bag_size=1000):
        self.forest = [DecisionTree() for i in range(forest)]
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

        prediction = np.zeros(testN, self.featureN)

        for idx, tree in enumerate(tqdm.tqdm(self.forest)):
            prediction[:,idx] = tree.predict(X)
        
        # Voting
        pred = np.argmax(prediction, axis=1)

        return pred 