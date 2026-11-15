"""
mtah_v2, three-state Viterbi decoding and the gradient boosted tree helpers shared by the training and decoding scripts
============================
viterbi3 decodes per frame H/A/D log probabilities into labels with one cost for each of three transitions (change of team,
into dead ball, out of dead ball), run inside each block; viterbi3_tv lets the transition costs vary per frame with the
probability of a nearby change point. boundary_labels builds the targets of the change-point model, fit and fit_two_stage
train the histogram gradient boosted tree classifiers of sklearn, spell_weights weights frames by the length of their
possession spell, and to_hadx converts class indices to the internal label codes.

Inputs  numpy arrays (see the docstring of each function)
Outputs  decoded labels, targets, fitted classifiers or weights

Usage
  from mtah_v2 import viterbi3, viterbi3_tv, boundary_labels, fit, to_hadx

Last modified 2026-11-15
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def viterbi3(logp, blocks, c_team, c_dead, c_alive):
    """Three-state Viterbi (0 H, 1 A, 2 D), run inside each block. logp (T, 3)"""
    cost = np.array([[0, c_team, c_dead], [c_team, 0, c_dead], [c_alive, c_alive, 0]], float)
    out = np.zeros(len(logp), np.int8)
    for a, b in blocks.values():
        n = b - a + 1
        lp = logp[a:b + 1]
        back = np.zeros((n, 3), np.int8)
        score = lp[0].copy()
        for t in range(1, n):
            cand = score[:, None] - cost          # (from, to)
            back[t] = cand.argmax(0)
            score = cand.max(0) + lp[t]
        s = int(score.argmax())
        for t in range(n - 1, -1, -1):
            out[a + t] = s
            s = back[t, s]
    return out


def viterbi3_tv(logp, blocks, c_team, c_dead, c_alive, pb=None, k=0.0, p0=0.04):
    """Three-state Viterbi with transition costs that vary per frame. pb is the probability that a boundary is near the frame, k is the strength,
    cost = base cost − k·(logit(pb) − logit(p0)), floored at 0.1. Same as viterbi3 when pb is None or k is 0"""
    base = np.array([[0, c_team, c_dead], [c_team, 0, c_dead], [c_alive, c_alive, 0]], float)
    if pb is None or k == 0:
        return viterbi3(logp, blocks, c_team, c_dead, c_alive)
    p = np.clip(pb, 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p)) - np.log(p0 / (1 - p0))
    off = (base > 0).astype(float)
    out = np.zeros(len(logp), np.int8)
    for a, b in blocks.values():
        n = b - a + 1
        lp = logp[a:b + 1]
        zz = z[a:b + 1]
        back = np.zeros((n, 3), np.int8)
        score = lp[0].copy()
        for t in range(1, n):
            cost = np.maximum(base - k * zz[t] * off, 0.1 * off)
            cand = score[:, None] - cost
            back[t] = cand.argmax(0)
            score = cand.max(0) + lp[t]
        s = int(score.argmax())
        for t in range(n - 1, -1, -1):
            out[a + t] = s
            s = back[t, s]
    return out


def boundary_labels(gt, blocks, halfwidth=6, kind="all"):
    """1 for frames within halfwidth frames of the nearest true change of possession or dead/live switch, 0 otherwise, -1 outside the blocks and for unlabelled frames.
    kind is all (both kinds of switch, the default), team (direct changes between home and away only) or dead (into and out of dead ball only)"""
    y = np.full(len(gt), -1, np.int8)
    for a, b in blocks.values():
        w = gt[a:b + 1]
        ok = w > 0
        m = (w[1:] != w[:-1]) & ok[1:] & ok[:-1]
        if kind == "team":
            m &= (w[:-1] != 3) & (w[1:] != 3)
        elif kind == "dead":
            m &= (w[:-1] == 3) | (w[1:] == 3)
        tr = np.flatnonzero(m) + 1
        lab = np.zeros(len(w), np.int8)
        for t in tr:
            lab[max(t - halfwidth, 0):t + halfwidth + 1] = 1
        lab[~ok] = -1
        y[a:b + 1] = lab
    return y


HGB_DEFAULT = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0)


def fit(X, y, seed=0, weight=None, params=None):
    """params overrides the hyperparameters in HGB_DEFAULT (used for hyperparameter sweeps; the fixed defaults apply when it is not given)"""
    clf = HistGradientBoostingClassifier(**{**HGB_DEFAULT, **(params or {})}, random_state=seed)
    clf.fit(X, y, sample_weight=weight)
    return clf


class TwoStageHGB:
    """Two-stage gradient boosted tree model. Stage one separates dead ball from live ball on all labelled frames, stage two separates home team from away team on live ball frames only.
    predict_proba combines them into three columns [H, A, D], where p_D comes from stage one, p_H = (1 - p_D) * p(H|live ball) and p_A = (1 - p_D) * p(A|live ball).
    The input y uses the encoding of mtah_v3 (0 H, 1 A, 2 D). The interface matches HistGradientBoostingClassifier (fit, predict_proba, classes_)."""
    def __init__(self, seed=0, params=None):
        self.seed, self.params = seed, params
        self.classes_ = np.array([0, 1, 2])
    def _new(self):
        return HistGradientBoostingClassifier(**{**HGB_DEFAULT, **(self.params or {})}, random_state=self.seed)
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.dead = self._new().fit(X, (y == 2).astype(int), sample_weight=sample_weight)
        m = y < 2
        self.team = self._new().fit(X[m], y[m], sample_weight=None if sample_weight is None else sample_weight[m])
        assert list(self.team.classes_) == [0, 1], self.team.classes_
        return self
    def predict_proba(self, X):
        pd_ = self.dead.predict_proba(X)[:, 1]
        pt = self.team.predict_proba(X)
        out = np.empty((len(pd_), 3), dtype=np.float64)
        out[:, 0] = (1.0 - pd_) * pt[:, 0]
        out[:, 1] = (1.0 - pd_) * pt[:, 1]
        out[:, 2] = pd_
        return out


def fit_two_stage(X, y, seed=0, weight=None, params=None):
    """fit for the two-stage gradient boosted tree model, same signature as fit"""
    return TwoStageHGB(seed=seed, params=params).fit(X, y, sample_weight=weight)


def spell_weights(gt, blocks, alpha=1.0, ref=250.0, lo=0.2, hi=20.0):
    """Weight every frame by the length of the spell it belongs to, (ref/spell length)^alpha, clipped to [lo, hi], then normalised to mean 1.
    spellF1 counts every spell once, but per frame training lets the long spells dominate and the short ones are hardly learnt. This weight gives every spell a similar share of the training.
    Unlabelled frames get weight 0"""
    w = np.zeros(len(gt), float)
    for a, b in blocks.values():
        seg = gt[a:b + 1]
        br = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.r_[0, br, len(seg)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if seg[s] > 0:
                w[a + s:a + e] = min(max((ref / (e - s)) ** alpha, lo), hi)
    m = w > 0
    if m.any():
        w[m] /= w[m].mean()
    return w


def to_hadx(codes3):
    """Convert 0 H, 1 A, 2 D to the internal codes 1 H, 2 A, 3 D"""
    return (codes3 + 1).astype(np.int8)
