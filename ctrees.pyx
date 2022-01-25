#cython: language_level=3

import logging
import time

import numpy as np
import pandas as pd

class Tree():

    class Node():
        def __init__(self, df, target):
            """Instantiates a root node."""
            self.idx = 0        # Root node is at index 0
            self.level = 0      # Top level of tree (root node) is level 0
            self.parent = None  # Index position of parent node
            self.df = df        # Pandas DataFrame attached to node
            self.size = self.df.shape[0]
            self.column = None  # String column name
            self.type = None    # "categorical" or "numeric"
            self.boundary = None   # Value at which DF will be split.
            self.target = target  # String, name of target variable in df
            self.pred = self.df[self.target].mean()
            self.left_child = None   # Index of left child
            self.right_child = None  # Index of right child
            self.right = None
            self.leaf = True   # True if node is a leaf node

        def __str__(self):
            if self.pred is not None:
                pred = "{:.1f}".format(self.pred)
            else:
                pred = None
            return (
                f"Node: {self.idx}, Level: {self.level}, Parent: {self.parent}, "
                f"Size: {self.size}\n"
                f"\tColumn: {self.column}, Type: {self.type}, "
                f"Boundary: {self.boundary}\n"
                f"\tTarget: {self.target}, Pred: {pred}\n"
                f"\tLeft: {self.left_child}, Right: {self.right_child}, Leaf: {self.leaf}"
            )

    class Split():
        def __init__(self, column, col_type):
            self.type = col_type
            self.column = column
            self.boundary = None
            self.RSS = None
            self.left_pred = None
            self.right_pred = None
            self.left = None
            self.right = None

        def __str__(self):
            preds = [None, None]
            for idx, pred in enumerate([self.left_pred, self.right_pred]):
                if pred is not None:
                    preds[idx] = "{:.1f}".format(pred)
                else:
                    preds[idx] = None
            return (
                f"Type: {self.type}, Column: {self.column}, "
                f"Boundary: {self.boundary}, RSS: {self.RSS:.3e}\n"
                f"\tLeft Size: {self.left.shape[0]}, "
                f"Left Prediction: {preds[0]}\n"
                f"\tRight Size: {self.right.shape[0]}, "
                f"Right Prediction: {preds[1]}"
            )


    def __init__(self, max_levels=None, min_leaf_size=5, verbose=False):
        """A decision tree object.

        Args:
            max_levels: The will stop training when this level is
                reached. The root node of the tree is at level 0, so
                a tree that stops at level 1 has one split.
            min_leaf_size: Leaves are not allowed to have fewer than
                this number of records. 
        """
        self.max_levels = max_levels
        self.min_leaf_size = min_leaf_size
        self.verbose = verbose
        self.tree = []

        logging.basicConfig(filename="training.log",
                            encoding="utf-8",
                            level=logging.DEBUG,
                            format="%(asctime)s:: %(message)s")

    def _info(self, msg):
        logging.info(msg)
        if self.verbose:
            print(msg)


    def _initialize_tree(self, df, target):
        self.tree = [self.Node(df, target)]

    def split_col(self, node, column):
        col = node.df[column]
        col_vals = np.sort(col.unique())
        if len(col_vals) < 2 or node.size < (2 * self.min_leaf_size):
            return None
        col_type = "categorical" if col.dtype == "O" else "numeric"
        min_split = self.Split(column, col_type)
        for boundary in col_vals[:-1]:
            if col_type == "categorical":
                left = node.df[col == boundary]
                right = node.df[col != boundary]
            else:
                left = node.df[col <= boundary]
                right = node.df[col > boundary]
            if (left.shape[0] < self.min_leaf_size or
                right.shape[0] < self.min_leaf_size):
                continue        
            left_mean = left[node.target].mean() # Ignores NA vals

            right_mean = right[node.target].mean()

            RSS = (sum((left[node.target] - left_mean)**2) + 
                   sum((right[node.target
                   ] - right_mean)**2) )
            if min_split.RSS is None or RSS < min_split.RSS:
                min_split.RSS = RSS
                min_split.boundary = boundary
                min_split.left = left
                min_split.right = right
                min_split.left_pred = left_mean
                min_split.right_pred = right_mean

        return min_split if min_split.RSS is not None else None

    def split_node(self, node):
        min_split = None

        for column in node.df.columns:
            if column == node.target:
                continue
            split = self.split_col(node, column)
            if split is None:
                continue
            if (min_split is None or
                split.RSS < min_split.RSS):
                min_split = split
        if min_split is not None:
            node.column = min_split.column
            node.type = min_split.type
            node.boundary = min_split.boundary
            # node.pred = None
            node.leaf = False
            left_child = self.create_child_node(node, min_split)
            right_child = self.create_child_node(node, min_split, right=True)
            return left_child, right_child
        else:
            return None

    @classmethod
    def create_child_node(cls, parent, split, right=False):
        if right:
            child = cls.Node(split.right, parent.target)
            child.pred = split.right_pred
        else:
            child = cls.Node(split.left, parent.target)
            child.pred = split.left_pred
        child.idx = None
        child.right = right
        child.level = parent.level + 1
        child.parent = parent.idx
        return child

    def append_node(self, node):
        node.idx = len(self.tree)
        if node.right:
            self.tree[node.parent].right_child = node.idx
        else:
            self.tree[node.parent].left_child = node.idx
        self.tree.append(node)

    def train(self, df, target):
        """Trains the decision tree.

        Args:
            df: A Pandas DataFrame containing the training data.
            target: A string with the name of the dataframe's target column.
        """
        self._initialize_tree(df, target)
        node_idx = 0
        start_time = time.time()
        while node_idx < len(self.tree):
            node = self.tree[node_idx]
            node_idx += 1
            # Do not exceed max number of tree levels
            if self.max_levels is not None and node.level >= self.max_levels:
                break
            if node.size < (2 * self.min_leaf_size):
                continue
            self._info(f"Splitting - level {node.level}, "
                       f"node {node.idx}, size {node.size}")
            child_nodes = self.split_node(node)
            # Continue to next node if tree cannot be split.
            if child_nodes is None:
                continue
            else:
                left_child, right_child = child_nodes

            self.append_node(left_child)
            self.append_node(right_child)
        elapsed_minutes = (time.time() - start_time) / 60
        print("Training Complete")
        print("Elapsed Time (min): ", elapsed_minutes)

    def predict_from_series(self, series):
        node = self.tree[0]
        while not node.leaf:
            if node.type == "categorical":
                if series[node.column] == node.boundary:
                    node = self.tree[node.left_child]
                else:
                    node = self.tree[node.right_child]
            else:
                if series[node.column] <= node.boundary:
                    node = self.tree[node.left_child]
                else:
                    node = self.tree[node.right_child]
        return node.pred

    def predict(self, df):
        return df.apply(self.predict_from_series, "columns")



    def __str__(self):
        return "\n".join([str(x) for x in self.tree])

