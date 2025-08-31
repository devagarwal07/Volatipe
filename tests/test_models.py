from src.models.garch_model import GARCHEnsemble, GARCHSpec
import pandas as pd
import numpy as np


def test_garch_spec_init():
    spec = GARCHSpec(type='garch', p=1, q=1)
    assert spec.type == 'garch'


def test_garch_ensemble_fit_forecast():
    # synthetic returns
    np.random.seed(0)
    r = pd.Series(np.random.randn(300) / 100)
    specs = [GARCHSpec(type='garch'), GARCHSpec(type='egarch', o=1)]
    ens = GARCHEnsemble(specs).fit(r)
    f = ens.forecast(1)
    assert 'ensemble' in f
