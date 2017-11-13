import numpy as np


class MyDecisionTree:
    def __init__(self, min_leaf_size=1):
        self.min_leaf_size_ = min_leaf_size
        self.tree_ = None

    def predict(self, x_test):
        return self.tree_(x_test)

    def fit(self, X, y):
        self.tree_ = Tree(X, y, self.min_leaf_size_)


class Tree:
    def __init__(self, X, y, min_leaf_size):
        self.left_node_ = None
        self.right_node_ = None
        self.y_ = y

        if len(y) <= min_leaf_size:
            return

        self.feature_, self.border_ = find_best_split(X, y)
        if self.feature_ == None:
            return

        bool_mask = X[:, self.feature_] <= self.border_
        self.left_node_ = Tree(X[bool_mask], y[bool_mask], min_leaf_size)
        self.right_node_ = Tree(X[np.logical_not(bool_mask)],
                                y[np.logical_not(bool_mask)], min_leaf_size)

    def __call__(self, x_test):
        if self.left_node_ == None:
            return np.mean(self.y_)
        else:
            if x_test[self.feature_] <= self.border_:
                return self.left_node_(x_test)
            else:
                return self.right_node_(x_test)


# this function can be implemented more effectively.
def find_best_split(X, y):
    best_tss = np.sum(np.power(y - np.mean(y), 2))
    best_feature = None
    best_border = None

    for feature_number in range(X.shape[1]):
        feature = X[:, feature_number]

        for i in range(len(feature)):
            bool_mask = X[:, feature_number] <= feature[i]
            left_tss = np.sum(
                np.power(y[bool_mask] - np.mean(y[bool_mask]), 2))
            right_tss = np.sum(
                np.power(y[~bool_mask] - np.mean(y[~bool_mask]), 2))

            if left_tss + right_tss < best_tss:
                best_feature = feature_number
                best_border = feature[i]
                best_tss = left_tss + right_tss

    return best_feature, best_border
