
import numpy as np
import pandas as pd
import pickle
import pytest

import trees
import util

TARGET = "Rented Bike Count"

@pytest.fixture
def split():
    bikes = pd.read_csv("SeoulBikeData.csv")
    train, dev = util.split(bikes, [0.8, 0.2], seed=2022)
    return train, dev

@pytest.fixture
def train(split):
    return split[0]

@pytest.fixture
def dev(split):
    return split[1]

@pytest.fixture
def pre_trained():
    with open("trained-tree-100-250.pickle", "rb") as tfile:
        return pickle.load(tfile)

def test_nodes():
    print()

    col_list = [TARGET, "Temperature(°C)", "Holiday"]
    train_small = train[col_list]

    # Create tree
    btree = trees.Tree(2, 5)
    assert btree.max_levels == 2
    assert btree.min_leaf_size == 5
    assert not btree.tree

    # Test split_col
    btree._initialize_tree(train_small, TARGET)
    assert len(btree.tree) == 1
    root_node = btree.tree[0]
    assert isinstance(root_node, trees.Tree.Node)
    assert root_node.size == train_small.shape[0]
    assert root_node.pred == train_small[TARGET].mean()
    assert root_node.leaf

    split_numeric = btree.split_col(root_node, col_list[1])
    assert split_numeric.boundary == 12.0
    
    split_cat = btree.split_col(root_node, col_list[2])
    assert split_cat.left.shape[0] == 355

    # Test split_node
    left, right = btree.split_node(root_node)
    assert not root_node.leaf
    assert root_node.column == "Temperature(°C)"
    assert root_node.boundary == 12.0
    assert left.leaf
    assert left.size == 3275
    assert round(left.pred, 1) == 361.9
    assert right.leaf
    assert right.size == 3733
    assert round(right.pred, 1) == 995.0

def test_train_small(train):
    print()
    col_list = [TARGET, "Temperature(°C)", "Holiday"]
    train_small = train[col_list]
    btree = trees.Tree(3, 5)
    btree.train(train_small, TARGET)
    level_sums = [0] * 4
    for node in btree.tree:
        level_sums[node.level] += node.size
    assert level_sums == [7008] * 4
    print(btree)

def test_train_all(train):
    print()
    btree = trees.Tree(3, 5)
    btree.train(train, TARGET)

    print(btree)

def test_unbalanced(train):
    print()
    btree = trees.Tree(100, 1000)
    btree.train(train, TARGET)
    print(btree)

# def write_pretrained_tree(train):
#     btree = trees.Tree(100, 250)
#     btree.train(train, TARGET)
#     with open("trained-tree-100-250.pickle", "wb") as tfile:
#         pickle.dump(btree, tfile)

def test_series_prediction(dev, pre_trained):
    print()
    features = dev.iloc[1]
    pred = pre_trained.predict_from_series(features)
    print(f"Prediction: {pred:.1f}, Actual: {features[TARGET]}")
    print(features)

def test_df_prediction(dev, pre_trained):
    print()
    print(pre_trained.predict(dev))





        
