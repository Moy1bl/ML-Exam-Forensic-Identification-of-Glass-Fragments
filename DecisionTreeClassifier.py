import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, information_gain=None, val=None):
        # applies only to decision node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain

        # applies only to leaf node
        self.val = val


class Decision_Tree_Classifier:
    def __init__(self, min_sample_split = 4, max_depth = 10):

        """root initialization"""
        self.root = None

        '''two stopping conditions'''
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def tree_builder(self, data, actual_depth = 0):
        """# assigning all values beside the last column to X (features) only last column values to y (target)"""
        X = data[:,:-1]
        Y = data[:,-1]
        n_samples, n_features = X.shape ##here different

        if n_samples >= self.min_sample_split and actual_depth < self.max_depth:
            best_split = self.do_best_split(data, n_samples, n_features)
            if best_split["information_gain"] > 0:
                subtree_l = self.tree_builder(best_split["data_left"], actual_depth +1)
                subtree_r = self.tree_builder(best_split["data_right"], actual_depth + 1)
                return Node(best_split["feature_idx"], best_split["threshold"], subtree_l, subtree_r, best_split["information_gain"])

        leaf_val = self.calc_leaf_val(Y)
        # return leaf node
        return Node(val=leaf_val)


    def do_best_split(self,data, n_samples, n_features):

        """initializing best split as dictionary"""
        best_split = {}
        info_gain_initial = -float("inf")

        for feature_idx in range(n_features): ##here different
            feature_vals = data[:, feature_idx]
            p_thresholds = np.unique(feature_vals)

            for threshold in p_thresholds:
                data_l, data_r = self.do_split(data, feature_idx, threshold)

                if len(data_l) > 0 and len(data_r) > 0:
                    y, l_y, r_y = data[:,-1], data_l[:,-1], data_r[:,-1]
                    actual_info_gain = self.calc_info_gain(y, l_y, r_y, "gini")

                    if actual_info_gain > info_gain_initial:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["data_left"] = data_l
                        best_split["data_right"] = data_r
                        best_split["information_gain"] = actual_info_gain
                        info_gain_initial = actual_info_gain

        return best_split



    def do_split(self, data, feature_idx, threshold):
        """ function to split the data """

        data_l = np.array([row for row in data if row[feature_idx] <= threshold])
        data_r = np.array([row for row in data if row[feature_idx] > threshold])
        return data_l, data_r


    def calc_info_gain (self, parent, l_child, r_child, mode = "entropy"):
        weight_left = len(l_child)/ len(parent)
        weight_right = len(r_child) / len(parent)

        if mode == "gini":
            gain = self.gini_impurity(parent) - (weight_left * self.gini_impurity(l_child) + weight_right * self.gini_impurity(r_child))
        else:
            gain = self.entropy_impurity(parent) -  (weight_left + self.entropy_impurity(l_child) + weight_right* self.entropy_impurity(r_child))
        return gain

    def entropy_impurity(self, y):
        c_labels = np.unique(y)
        entropy = 0
        for clas in c_labels:
            p_class = len(y[y == clas]) / len(y)
            entropy += -p_class * np.log2(p_class)
        return entropy

    def gini_impurity(self, y):
        c_labels = np.unique(y)
        gini = 0
        for clas in c_labels:
            p_clas = len(y[y == clas]) / len(y)
            gini += p_clas ** 2
        return 1 - gini

    def calc_leaf_val(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        data = np.column_stack((X, Y))
        self.root = self.tree_builder(data)

    def predict(self, X):
        prediction = [self.make_predict(x, self.root) for x in X]
        return prediction

    def make_predict( self, x, tree):
        if tree.val is not None:
            return tree.val

        feature_values = x[tree.feature_idx]

        if feature_values <= tree.threshold:
            return self.make_predict(x, tree.left)
        else:
            return self.make_predict(x, tree.right)


    def print_tree( self, tree=None, indent=" " ):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.val is not None:
            print(tree.val)
        else:
            print("X_" + str(tree.feature_idx), "<=", tree.threshold, "?", tree.information_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)