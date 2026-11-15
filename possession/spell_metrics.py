"""
spell_metrics, our own implementation of macroF1 and spellF1 following the definitions of the official evaluation/possession_metrics.py
============================
Used to score the held-out public matches, because the official scorer checks the 199,319 frames of BAR-ATH.
The official script gives the final numbers. The definitions follow the official documentation and code. The key points are as follows.
  Labels H=0, A=1, D=2, X=3. Frames whose ground truth is X are not graded.
  The valid segments (runs) are the contiguous intervals where the ground truth is not X (for BAR-ATH, the two halves). All temporal quantities are computed within a segment, never across segments.
  X in the prediction is first forward filled within the segment (backward filled at the start of a segment) before the temporal quantities are computed; frame-level metrics use the unfilled X and count it as wrong.
  macroF1, the mean of the F1 of the three ground truth classes H/A/D; the prediction columns include X.
  spellF1, a possession spell is a maximal contiguous interval of H or A within a segment (D is a pause, not possession). The temporal IoU is computed for ground truth and predicted spells of the same team that overlap,
  they are matched one to one greedily from the largest IoU down, one F1 is computed for each IoU threshold 0.3, 0.4, 0.5, 0.6, 0.7, and the mean is taken.

Inputs  integer code arrays of the ground truth and of the prediction (same length)
Outputs  evaluate() returns a dictionary

Usage
  from possession.spell_metrics import codes_from_labels, evaluate
  r = evaluate(gt_codes, pred_codes); print(r["spell_f1"], r["macro_f1"])

Last modified 2026-11-15
"""
import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def codes_from_labels(labels):
    """Convert string labels (H/A/D/X, tolerant of case and whitespace) to integer codes"""
    s = np.char.upper(np.char.strip(np.asarray(labels, dtype=str)))
    codes = np.full(len(s), -1, np.int8)
    for k, name in enumerate(LABELS):
        codes[s == name] = k
    if (codes < 0).any():
        raise ValueError("unknown label found")
    return codes


def codes_from_hadx(lab):
    """Convert the internal codes of this project (0 none/X, 1 H, 2 A, 3 D) to the official codes"""
    lab = np.asarray(lab)
    return np.select([lab == 1, lab == 2, lab == 3], [H, A, D], X).astype(np.int8)


def valid_runs(gt):
    graded = (gt != X).astype(np.int8)
    e = np.diff(np.r_[0, graded, 0])
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def fill_abstentions(pred, runs):
    out = pred.copy()
    for a, b in runs:
        w = out[a:b]
        known = np.flatnonzero(w != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != X, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[a:b] = w[carry]
    return out


def segments(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.r_[0, br, len(w)]
        out.extend((a + int(bounds[i]), a + int(bounds[i + 1]), int(w[bounds[i]])) for i in range(len(bounds) - 1))
    return out


def spells(codes, runs):
    return [s for s in segments(codes, runs) if s[2] in (H, A)]


def spell_f1(gt_sp, pr_sp, ious=IOUS):
    ov = []
    for i, (ga, gb, gc) in enumerate(gt_sp):
        for j, (pa, pb, pc) in enumerate(pr_sp):
            if pc != gc or pb <= ga or gb <= pa:
                continue
            inter = min(gb, pb) - max(ga, pa)
            union = max(gb, pb) - min(ga, pa)
            ov.append((inter / float(union), i, j))
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    per = {}
    for thr in ious:
        used_g, used_p = set(), set()
        for iou, i, j in ov:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j)
        tp = len(used_g)
        fp, fn = len(pr_sp) - tp, len(gt_sp) - tp
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        per[thr] = 2 * p * r / (p + r) if p + r else 0.0
    return float(np.mean(list(per.values()))), per


def macro_f1(gt, pred):
    m = gt != X
    cm = np.zeros((3, 4), np.int64)
    np.add.at(cm, (gt[m].astype(int), pred[m].astype(int)), 1)
    f1s = []
    for k in range(3):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k].sum() - tp
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), f1s, cm


def evaluate(gt, pred, fps=25.0):
    gt, pred = np.asarray(gt, np.int8), np.asarray(pred, np.int8)
    if len(gt) != len(pred):
        raise ValueError("lengths differ")
    runs = valid_runs(gt)
    filled = fill_abstentions(pred, runs)
    gs, ps = spells(gt, runs), spells(filled, runs)
    sf1, per = spell_f1(gs, ps)
    mf1, f1s, cm = macro_f1(gt, pred)
    return {"spell_f1": sf1, "spell_f1_per_iou": per, "macro_f1": mf1, "f1_H": f1s[0], "f1_A": f1s[1], "f1_D": f1s[2],
            "n_gt_spells": len(gs), "n_pred_spells": len(ps),
            "gt_mean_spell_s": float(np.mean([b - a for a, b, _ in gs]) / fps) if gs else 0.0,
            "pred_mean_spell_s": float(np.mean([b - a for a, b, _ in ps]) / fps) if ps else 0.0,
            "n_graded": int((gt != X).sum()), "n_abstain_on_graded": int(((gt != X) & (pred == X)).sum()),
            "accuracy": float((gt[gt != X] == pred[gt != X]).mean()) if (gt != X).any() else 0.0}


def merge_short_spells(gt, blocks, min_frames):
    """Merge possession spells shorter than min_frames frames in the internal labels (0 none, 1 H, 2 A, 3 D) into the previous spell (only when the previous one is also a possession spell).
    Used to apply the same rule across sources for how short a spell is too short to count as a change of possession; 0 disables merging"""
    gt = np.asarray(gt).copy()
    if min_frames <= 0:
        return gt
    for a, b in blocks.values():
        w = gt[a:b + 1]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.r_[0, br, len(w)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if w[s] in (1, 2) and e - s < min_frames and i > 0 and w[bounds[i - 1]] in (1, 2):
                w[s:e] = w[bounds[i - 1]]
        gt[a:b + 1] = w
    return gt


def binned_hit_rates(gt_codes, pred_codes, fps=25.0, iou=0.5, bins=((0, 3), (3, 6), (6, 12), (12, 30), (30, 1e9))):
    """Hit rate grouped by the duration of the ground truth spell (one to one IoU matching); returns {"[0,3)": hit rate, ...} and the number of spells in each group"""
    gt_codes, pred_codes = np.asarray(gt_codes, np.int8), np.asarray(pred_codes, np.int8)
    runs = valid_runs(gt_codes)
    gs, ps = spells(gt_codes, runs), spells(fill_abstentions(pred_codes, runs), runs)
    ov = []
    for i, (ga, gb, gc) in enumerate(gs):
        for j, (pa, pb, pc) in enumerate(ps):
            if pc != gc or pb <= ga or gb <= pa:
                continue
            inter = min(gb, pb) - max(ga, pa)
            union = max(gb, pb) - min(ga, pa)
            ov.append((inter / float(union), i, j))
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    hit = np.zeros(len(gs), bool)
    used = set()
    for v, i, j in ov:
        if v < iou:
            break
        if hit[i] or j in used:
            continue
        hit[i] = True
        used.add(j)
    L = np.array([(b - a) / fps for a, b, _ in gs]) if gs else np.zeros(0)
    out = {}
    for lo, hi in bins:
        m = (L >= lo) & (L < hi)
        out["[%g,%g)" % (lo, hi)] = (float(hit[m].mean()) if m.any() else float("nan"), int(m.sum()))
    return out


def validate(frame_col, labels, n_expected):
    """The four checks of the official validate_submission; returns None or the reason"""
    if len(labels) != n_expected:
        return "row count %d != %d" % (len(labels), n_expected)
    f = np.asarray(frame_col)
    if not np.issubdtype(f.dtype, np.integer) or not np.array_equal(f, np.arange(n_expected)):
        return "frame column is not 0..%d in order" % (n_expected - 1)
    s = np.char.upper(np.char.strip(np.asarray(labels, dtype=str)))
    bad = sorted(set(s.tolist()) - set(LABELS.tolist()))
    if bad:
        return "labels outside HADX: %s" % bad[:5]
    return None
