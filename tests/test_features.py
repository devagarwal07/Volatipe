from src.features.build_features import parkinson, garman_klass
import pandas as pd
import numpy as np

def test_parkinson():
    high = pd.Series([10, 11, 12])
    low = pd.Series([9, 10, 11])
    out = parkinson(high, low)
    assert len(out) == 3
    assert np.all(out >= 0)

def test_garman_klass():
    o = pd.Series([10, 10, 10])
    h = pd.Series([11, 11, 11])
    l = pd.Series([9, 9, 9])
    c = pd.Series([10.5, 10.5, 10.5])
    out = garman_klass(o, h, l, c)
    assert len(out) == 3
