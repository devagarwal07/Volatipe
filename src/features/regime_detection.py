from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from ..utils.logging import get_logger

logger = get_logger(__name__)

REGIME_LABELS = {0: 'calm', 1: 'normal', 2: 'stressed', 3: 'crisis'}


def fit_hmm(returns: pd.Series, n_states: int = 4, seed: int = 42) -> GaussianHMM:
    model = GaussianHMM(n_components=n_states, covariance_type='full', random_state=seed, n_iter=200)
    X = returns.dropna().values.reshape(-1, 1)
    model.fit(X)
    return model


def classify_hmm(model: GaussianHMM, returns: pd.Series) -> pd.Series:
    X = returns.dropna().values.reshape(-1, 1)
    hidden = model.predict(X)
    mapped = pd.Series(hidden, index=returns.dropna().index).map(REGIME_LABELS)
    return mapped


def fit_gmm(features: pd.DataFrame, n_components: int = 4, seed: int = 42) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    X = features.dropna().values
    gmm.fit(X)
    return gmm


def classify_gmm(gmm: GaussianMixture, features: pd.DataFrame) -> pd.Series:
    probs = gmm.predict_proba(features.dropna().values)
    labels = probs.argmax(axis=1)
    series = pd.Series(labels, index=features.dropna().index).map(REGIME_LABELS)
    return series


def combine_regimes(hmm_reg: pd.Series, gmm_reg: pd.Series) -> pd.Series:
    combo = pd.concat([hmm_reg, gmm_reg], axis=1, keys=['hmm', 'gmm'])
    # simple heuristic: crisis if any crisis, stressed if any stressed, else majority
    def row_rule(r):
        vals = set(r.values)
        if 'crisis' in vals:
            return 'crisis'
        if 'stressed' in vals:
            return 'stressed'
        if 'normal' in vals:
            return 'normal'
        return 'calm'
    return combo.apply(row_rule, axis=1)
