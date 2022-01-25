import math
import random

import pandas as pd

def split(df, splits, seed=None):
    if sum(splits) > 1 or sum(splits) <= 0:
        raise(ValueError)
    if not math.isclose(sum(splits), 1):
        splits.append(1 - sum(splits))
    
    rows = df.shape[0]
    split_labels = list(range(len(splits)))
    split_sizes = [int(round(rows * split)) for split in splits[:-1]]
    split_sizes.append(rows - sum(split_sizes))
    classes = [[label] * split_size
               for label, split_size in zip(split_labels, split_sizes)]
    classes = [x for split_class in classes for x in split_class]
    classes = pd.Series(classes)
    random.seed(seed)
    random.shuffle(classes)

    dfs = [df[classes == x] for x in split_labels]
    return tuple(dfs)