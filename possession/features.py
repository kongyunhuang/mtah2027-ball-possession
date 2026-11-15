"""
features, time-window helpers used by possession/mtah_features.py
============================
seg_bounds splits a block or period array into contiguous runs, and window_mean takes the mean of a per frame series over the w frames before
(or, with future, after) every frame, without crossing a run boundary.

Inputs  per frame numpy arrays and the per frame block or period array seg
Outputs  numpy arrays of the same length

Usage
  from possession.features import window_mean
  m = window_mean(x, 25, M["period"])

Last modified 2026-11-15
"""
import numpy as np


def seg_bounds(seg):
    starts = np.r_[0, np.where(seg[1:] != seg[:-1])[0] + 1]
    return list(zip(starts, list(starts[1:]) + [len(seg)]))


def window_mean(x, w, seg, future=False):
    out = np.zeros(len(x))
    for s0, s1 in seg_bounds(seg):
        cs = np.r_[0, np.cumsum(x[s0:s1])]
        i = np.arange(s1 - s0)
        if future:
            hi = np.minimum(i + 1 + w, s1 - s0)
            out[s0:s1] = (cs[hi] - cs[i + 1]) / np.maximum(hi - i - 1, 1)
        else:
            lo = np.maximum(i - w, 0)
            out[s0:s1] = (cs[i] - cs[lo]) / np.maximum(i - lo, 1)
    return out
