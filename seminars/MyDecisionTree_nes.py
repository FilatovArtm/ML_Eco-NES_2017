class MyDecisionTree:
    def __init__(self, min_leaf_size=1):
        self.min_leaf_size_ = min_leaf_size
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = Tree(X, y, self.min_leaf_size_)

    def predict(self, X):
        return self.tree_(X)

class Tree:
    def __init__(self, X, y, min_leaf_size):
        self.y_ = y
        self.left_child = None
        self.right_child = None

        if len(y) <= min_leaf_size:
            return

        self.feature, self.border = find_best_split(X, y)
        if self.feature == None:
            return

        split_mask = X[:, self.feature] <= self.border

        self.left_child = Tree(X[split_mask], y[split_mask], min_leaf_size)
        self.right_child = Tree(X[np.logical_not(split_mask)],
                                y[np.logical_not(split_mask)], min_leaf_size)

    def __call__(self, X):
        if self.left_child == None:
            return np.zeros(len(X)) + np.mean(self.y_)

        split_mask = X[:, self.feature] <= self.border
        predictions = np.zeros(len(X))
        predictions[split_mask] = self.left_child(X[split_mask])
        predictions[np.logical_not(split_mask)] = \
                    self.right_child(X[np.logical_not(split_mask)])
        return predictions

def find_best_split(X, y):
    best_error = np.sum(np.power(y - np.mean(y), 2))
    best_border = None
    best_feature = None

    for feature in range(X.shape[1]):
        for i in range(X.shape[0]):
            split_mask = X[:, feature] <= X[i, feature]
            error = y[split_mask] - np.mean(y[split_mask])
            left_error = np.sum(np.power(error, 2))

            error = y[np.logical_not(split_mask)] - \
                    np.mean(y[np.logical_not(split_mask)])
            right_error = np.sum(np.power(error, 2))
            if left_error + right_error < best_error:
                best_feature = feature
                best_border = X[i, feature]
                best_error = left_error + right_error

    return best_feature, best_border
