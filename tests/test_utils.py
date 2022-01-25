import pandas as pd

import util

def test_split():
    bikes = pd.read_csv("SeoulBikeData.csv")
    assert bikes.shape == (8760, 14)
    train, test, dev = util.split(bikes, [0.6, 0.2])

    assert train.shape == (int(round(8760 * 0.6)), 14)
    assert test.shape == (int(round(8760 * 0.2)), 14)
    assert dev.shape == (8760 - (train.shape[0] + test.shape[0]), 14)