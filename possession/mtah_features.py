"""
mtah_features, per-frame features that do not rely on player identity and tolerate a missing ball, shared by the public training matches and BAR-ATH (prediction)
============================
The input is the unified structure from the adapters (ball_xy is NaN when missing, player_xy per slot, slot_team, blocks). All forward and backward filling is done within a graded block,
never across blocks. The task is offline, so future frames may be used. The base features fall into five groups; the optional groups (gap, touch, kinematic, time-to-arrival,
positioning, team shape, pressing, roster, direction, co-movement and per-player features) are switched on by the arguments of build_features and described in their functions.
  Ball. Visibility, frames since it was last seen and until it is next seen, smoothed displacement speed over 1 s and 0.2 s, whether it is inside the pitch, distance to the touchlines and goal lines, whether it is in a penalty area.
  Player and ball. Relative to "the current or most recently seen ball position", the distances of the two nearest players of each team, the number of players within 3 m and 8 m, which team is nearest (only when the ball is visible).
  Shape of the two teams. For each team the number of players on the pitch, the centroid, the spread, the mean 1 s displacement speed and the share of running players, plus the differences between the two teams.
  Attacking direction. The attacking direction of the home team is inferred from the position of the rearmost player of each team in the block, and the x of the ball and of the centroids is converted to "home team attacking direction" coordinates.
  Time windows. Means of a few key quantities over the past and future 1 s, 3 s and 10 s.

Inputs  M (see adapters/mtah2027.py)
Outputs  X (T, F) float32, and the list names

Usage
  from possession.mtah_features import build_features
  X, names = build_features(M)

Last modified 2026-11-15
"""
import numpy as np
import pandas as pd

from .features import window_mean


def _smooth(xy, w):
    if w <= 1:
        return xy
    return pd.DataFrame(xy).rolling(w, center=True, min_periods=3).mean().values


def _ffill_bfill(xy, blocks, T):
    """Forward fill and backward fill within each block; returns (last_xy, age_last, next_xy, age_next), with NaN and 10**6 where there is nothing to fill from"""
    last, nxt = np.full_like(xy, np.nan), np.full_like(xy, np.nan)
    age_l, age_n = np.full(T, 10 ** 6), np.full(T, 10 ** 6)
    for a, b in blocks.values():
        w = xy[a:b + 1]
        ok = ~np.isnan(w[:, 0])
        idx = np.arange(len(w))
        j = np.maximum.accumulate(np.where(ok, idx, -1))
        has = j >= 0
        last[a:b + 1][has] = w[j[has]]
        age_l[a:b + 1][has] = (idx - j)[has]
        jn = np.minimum.accumulate(np.where(ok, idx, 10 ** 9)[::-1])[::-1]
        hasn = jn < 10 ** 9
        nxt[a:b + 1][hasn] = w[jn[hasn]]
        age_n[a:b + 1][hasn] = (jn - idx)[hasn]
    return last, age_l, nxt, age_n


def _disp_speed(xy, lag, fps):
    v = np.full(len(xy), np.nan)
    v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=1) * fps / lag
    return v


def _attack_dir(M, blocks):
    """Attacking direction of the home team per block, +1 towards +x and -1 towards -x. Uses the median position of the rearmost player of each team; the home team attacks +x if its rearmost player is closer to x=0"""
    L = M["pitch"][0]
    P = M["player_xy"]
    st = M["slot_team"]
    out = {}
    for h, (a, b) in blocks.items():
        hx = P[a:b + 1][:, st == 1, 0]
        ax = P[a:b + 1][:, st == 2, 0]
        with np.errstate(all="ignore"):
            h_min, h_max = np.nanmedian(np.nanmin(hx, 1)), np.nanmedian(np.nanmax(hx, 1))
            a_min, a_max = np.nanmedian(np.nanmin(ax, 1)), np.nanmedian(np.nanmax(ax, 1))
        # when the home team attacks +x, its rearmost player is on the low x side and the rearmost away player on the high x side
        score_plus = (L - a_max) + h_min
        score_minus = a_min + (L - h_max)
        out[h] = 1 if score_plus < score_minus else -1
    return out


def _nearest_slot(P, t, xy, team=None, st=None):
    """Slot nearest to xy in frame t (optionally restricted to one team); returns -1 if there is none"""
    d = np.linalg.norm(P[t] - xy[None, :], axis=1)
    if team is not None:
        d = np.where(st == team, d, np.inf)
    d = np.where(np.isnan(d), np.inf, d)
    i = int(np.argmin(d))
    return i if np.isfinite(d[i]) else -1


def _follow(P, st, s, e, slot, xy0, forward=True):
    """Follow one slot from s to e (half open); when the slot drops out, switch to the nearest player of the same team. Returns the positions (e-s, 2) and a 0/1 flag telling whether the original slot is still present"""
    n = e - s
    pos = np.full((n, 2), np.nan)
    alive = np.ones(n)
    idx = np.arange(s, e) if forward else np.arange(e - 1, s - 1, -1)
    team = st[slot] if slot >= 0 else None
    cur, last_xy, swapped = slot, xy0.copy(), False
    k = 0
    while k < n:
        t = idx[k]
        if cur >= 0 and not np.isnan(P[t, cur, 0]):
            last_xy = P[t, cur]
        else:
            cur = _nearest_slot(P, t, last_xy, team, st) if team is not None else -1
            swapped = True
            if cur >= 0:
                last_xy = P[t, cur]
        pos[k] = last_xy
        alive[k] = 0.0 if swapped else 1.0
        k += 1
    if not forward:
        pos, alive = pos[::-1], alive[::-1]
    return pos, alive


def gap_features(M, A_ref, F, sxy, seg, blocks, fps):
    """Features describing at whose feet the ball is inside a ball gap. Outside gaps the values are -1 and in_gap is 0.
    The function also adds 4 team shape dynamics columns that are defined on every frame, d_spread_h_1s and d_spread_a_1s (1 s change of the spread of each team) and
    v_centroid_h_dir and v_centroid_a_dir (1 s change of the centroid of each team along the home attacking direction)"""
    T = len(seg)
    L, W = M["pitch"]
    P = M["player_xy"].astype(np.float32)
    st = M["slot_team"]
    braw = M["ball_xy"]                      # raw ball positions are used for the first and last frame of a gap and for anchoring (smoothing eats about 3 frames at each end of a gap)
    vis = ~np.isnan(braw[:, 0])
    hd = A_ref["hd"]
    names = ["in_gap", "gap_pos", "last_team_nearest", "last_d_nearest", "last_ball_speed", "last_heading_out", "next_team_nearest",
             "next_d_nearest", "next_ball_speed", "ends_same_team", "path_x_dir", "path_y", "follow_x_dir", "follow_y", "follow_slot_alive",
             "backfollow_x_dir", "backfollow_y", "disagree_path_follow", "disagree_follow_back", "dh1_path", "da1_path", "nh3_path", "na3_path",
             "dh1_follow", "da1_follow", "carrier_opp_dist", "t_duel_rel", "d_duel", "before_duel", "carrier_team_infer", "carrier_conf", "infer_x_gap_pos",
             "anc_start_team", "anc_start_rel", "anc_end_team", "anc_end_rel", "unknown_span_s", "anchors_same_team"]
    G = {k: np.full(T, -1.0) for k in names}
    G["in_gap"][:] = 0.0

    def raw_vel(t, forward):
        """Ball velocity vector at frame t, from the displacement of the raw ball position over the 3 frames before (or after); falls back to the smoothed displacement when that is unavailable"""
        if forward and t - 3 >= 0 and vis[t - 3] and vis[t]:
            return (braw[t] - braw[t - 3]) * (fps / 3)
        if not forward and t + 3 < T and vis[t + 3] and vis[t]:
            return (braw[t + 3] - braw[t]) * (fps / 3)
        if t - 1 >= 0 and not np.isnan(sxy[t, 0]) and not np.isnan(sxy[t - 1, 0]):
            return (sxy[t] - sxy[t - 1]) * fps
        return np.zeros(2)

    def flight_anchor(t0, vvec, s, e, forward):
        """When the ball is in flight, extrapolate along its velocity (linear deceleration, within 2 s) and return the team of the first player the ball passes within 1.5 m and the step at which that happens; returns (0, -1) if there is none"""
        speed = float(np.linalg.norm(vvec))
        if speed < 5.0:
            return 0, -1
        for dk in range(1, 51):
            t = t0 + dk if forward else t0 - dk
            if not (s <= t < e):
                break
            pos = braw[t0] + vvec * (dk / fps) * (1.0 - 0.5 * dk / 50.0) * (1 if forward else -1)
            c = _nearest_slot(P, t, pos)
            if c >= 0 and np.linalg.norm(P[t, c] - pos) < 1.5:
                return int(st[c]), dk
        return 0, -1
    for a, b in blocks.values():
        v = vis[a:b + 1]
        edges = np.diff(np.r_[1, v.astype(np.int8), 1])
        starts = np.flatnonzero(edges == -1) + a
        ends = np.flatnonzero(edges == 1) + a       # half open, the gap is [s, e)
        for s, e in zip(starts, ends):
            if s <= a or e > b or e - s < 2:
                continue                          # both ends must be visible inside the block
            tl, tn = s - 1, e
            n = e - s
            xy0, xy1 = braw[tl], braw[tn]
            c0 = _nearest_slot(P, tl, xy0)
            c1 = _nearest_slot(P, tn, xy1)
            T0 = int(st[c0]) if c0 >= 0 else 0
            T1 = int(st[c1]) if c1 >= 0 else 0
            d0 = float(np.linalg.norm(P[tl, c0] - xy0)) if c0 >= 0 else 99.0
            d1 = float(np.linalg.norm(P[tn, c1] - xy1)) if c1 >= 0 else 99.0
            v0vec, v1vec = raw_vel(tl, True), raw_vel(tn, False)
            v0, v1 = float(np.linalg.norm(v0vec)), float(np.linalg.norm(v1vec))
            # if the ball is in flight, extrapolate to find at whose feet it lands (start of the gap) and who kicked it (end of the gap); only the interval between the two anchors is truly unknown
            Ta, dka = flight_anchor(tl, v0vec, s, e, True)
            Tb, dkb = flight_anchor(tn, v1vec, s, e, False)
            k_lo = dka if dka > 0 else 0                 # start and end of the unknown interval inside the gap (relative to the first gap frame)
            k_hi = n - dkb if dkb > 0 else n
            if k_hi <= k_lo:
                k_lo, k_hi = 0, n
            T0e = Ta if Ta > 0 else T0                    # team at each end after anchoring
            T1e = Tb if Tb > 0 else T1
            # seconds until the ball goes out of play
            vx, vy = v0vec
            tx = (L - xy0[0]) / vx if vx > 1e-3 else (-xy0[0] / vx if vx < -1e-3 else np.inf)
            ty = (W - xy0[1]) / vy if vy > 1e-3 else (-xy0[1] / vy if vy < -1e-3 else np.inf)
            tout = min(tx, ty)
            heading_out = 1.0 if 0 <= tout <= 3.0 else 0.0
            gp = (np.arange(n) + 1) / float(n + 1)
            path = xy0[None, :] + (xy1 - xy0)[None, :] * gp[:, None]
            follow, alive = _follow(P, st, s, e, c0, np.asarray(xy0), True) if c0 >= 0 else (path.copy(), np.zeros(n))
            back, _ = _follow(P, st, s, e, c1, np.asarray(xy1), False) if c1 >= 0 else (path.copy(), np.zeros(n))
            follow = np.where(np.isnan(follow), path, follow)
            back = np.where(np.isnan(back), path, back)
            sl = slice(s, e)
            G["in_gap"][sl] = 1.0
            G["gap_pos"][sl] = gp
            G["last_team_nearest"][sl], G["last_d_nearest"][sl], G["last_ball_speed"][sl], G["last_heading_out"][sl] = T0, d0, v0, heading_out
            G["next_team_nearest"][sl], G["next_d_nearest"][sl], G["next_ball_speed"][sl] = T1, d1, v1
            G["ends_same_team"][sl] = 1.0 if (T0 == T1 and T0 > 0) else 0.0
            hdir = hd[s:e]
            G["path_x_dir"][sl] = np.where(hdir > 0, path[:, 0], L - path[:, 0])
            G["path_y"][sl] = path[:, 1]
            G["follow_x_dir"][sl] = np.where(hdir > 0, follow[:, 0], L - follow[:, 0])
            G["follow_y"][sl] = follow[:, 1]
            G["follow_slot_alive"][sl] = alive
            G["backfollow_x_dir"][sl] = np.where(hdir > 0, back[:, 0], L - back[:, 0])
            G["backfollow_y"][sl] = back[:, 1]
            G["disagree_path_follow"][sl] = np.linalg.norm(path - follow, axis=1)
            G["disagree_follow_back"][sl] = np.linalg.norm(follow - back, axis=1)
            Pg = P[s:e]
            for ref, tag in ((path, "path"), (follow, "follow")):
                d = np.linalg.norm(Pg - ref[:, None, :].astype(np.float32), axis=2)
                d = np.where(np.isnan(d), 99.0, d)
                dh = np.where(st[None, :] == 1, d, 99.0)
                da = np.where(st[None, :] == 2, d, 99.0)
                G["dh1_" + tag][sl] = dh.min(1)
                G["da1_" + tag][sl] = da.min(1)
                if tag == "path":
                    G["nh3_path"][sl] = (dh < 3).sum(1)
                    G["na3_path"][sl] = (da < 3).sum(1)
            # anchor features
            G["anc_start_team"][sl] = Ta
            G["anc_start_rel"][sl] = (dka / float(n)) if dka > 0 else -1.0
            G["anc_end_team"][sl] = Tb
            G["anc_end_rel"][sl] = (1.0 - dkb / float(n)) if dkb > 0 else -1.0
            G["unknown_span_s"][sl] = (k_hi - k_lo) / fps
            G["anchors_same_team"][sl] = 1.0 if (T0e == T1e and T0e > 0) else 0.0
            # duel point, searched only inside the unknown interval, distance from the trajectory of the followed player to the nearest opponent
            if T0e in (1, 2):
                d = np.linalg.norm(Pg - follow[:, None, :].astype(np.float32), axis=2)
                d = np.where(np.isnan(d), 99.0, d)
                dopp = np.where(st[None, :] == 3 - T0e, d, 99.0).min(1)
                dopp = np.where(dopp >= 99.0, np.nan, dopp)
                win = dopp[k_lo:k_hi]
                if np.isfinite(win).any():
                    k = k_lo + int(np.nanargmin(win))
                    G["carrier_opp_dist"][sl] = np.nan_to_num(dopp, nan=-1.0)
                    G["t_duel_rel"][sl] = (k + 1) / float(n + 1)
                    G["d_duel"][sl] = float(dopp[k])
                    G["before_duel"][sl] = (np.arange(n) < k).astype(float)
                    dd = float(dopp[k])
                else:
                    k, dd = (k_lo + k_hi) // 2, 99.0
            else:
                k, dd = (k_lo + k_hi) // 2, 99.0
            # inference
            conf = 1.0
            if (Ta == 0 and d0 > 3.0) or (Tb == 0 and d1 > 3.0):
                conf *= 0.5
            if T0e > 0 and T0e == T1e and dd > 2.0:
                team = np.full(n, T0e, float)
                conf *= 0.9
            elif T0e > 0 and T1e > 0 and T0e != T1e:
                team = np.where(np.arange(n) < k, T0e, T1e).astype(float)
                conf *= 0.5
            elif T0e > 0 and T0e == T1e:
                team = np.full(n, T0e, float)
                conf *= 0.6
            else:
                team = np.zeros(n)
                conf = 0.0
            G["carrier_team_infer"][sl] = team
            G["carrier_conf"][sl] = conf
            G["infer_x_gap_pos"][sl] = np.where(team == 1, 1.0, np.where(team == 2, -1.0, 0.0)) * gp
    for k_, v_ in G.items():
        v_ = np.nan_to_num(v_, nan=-1.0, posinf=99.0, neginf=-99.0)
        F["gap_" + k_] = v_.astype(np.float32)
    # team shape dynamics, also meaningful outside gaps
    for side in ("h", "a"):
        sp = F["spread_" + side]
        ds = np.zeros(T)
        lag = int(round(fps))
        ds[lag:] = sp[lag:] - sp[:-lag]
        F["d_spread_%s_1s" % side] = ds
        cx = F["c%s_x_dir" % side]
        vc = np.zeros(T)
        vc[lag:] = cx[lag:] - cx[:-lag]
        F["v_centroid_%s_dir" % side] = vc



TOUCH_SETS = {"robust": dict(angle_deg=45.0, dv=5.0, r=2.0, k=8, nms=6, min_speed=3.0, back_max=25),     # for tracking with about 0.3 m of position noise
              "clean": dict(angle_deg=30.0, dv=3.0, r=2.0, k=4, nms=6, min_speed=1.5, back_max=25)}     # for low-jitter tracking such as the competition tracking (ball jitter about 0.01 m)
# r is the assignment radius and back_max the maximum number of frames to step back along the velocity when the assignment finds no player. The radius of 2 m and the step back of 25 frames were set while
# checking that precision and recall against the event table of the Metrica sample data stay unchanged and that the share of touches assigned to a player on the competition tracking rises (label-free)


def touch_frame_features(M, F, blocks, fps, bxy, cap_s=30.0, touch_set="robust"):
    """Touch sequence features. Touches are found by touch_events.detect_touches with the threshold set given by touch_set (robust or clean, see TOUCH_SETS; the robust set, angle 45, speed difference 5, k 8,
    minimum speed 3, was checked against the Metrica event table, precision 0.67 to 0.72 and recall 0.83 to 0.85 with 0.3 m of added noise). Each frame gets the team of the last and of the next owned touch, the seconds since and until them,
    whether the two are the same team, the number of consecutive touches by the same team, the player-ball distance of the last touch, the ball speed, the score, whether it was a strong touch (speed change kind) and the distance travelled by the ball since the last touch; strong touches get a set of their own;
    unowned touches only get the seconds since the last one; the number of owned touches of each team in the past and future 2 s is added as well. The value is -1 outside blocks and where nothing is available"""
    from possession.touch_events import detect_touches
    T = len(bxy)
    touches = detect_touches(M, **TOUCH_SETS[touch_set])
    names = ["t_last_team", "t_last_age_s", "t_next_team", "t_next_in_s", "t_same_team", "t_run_same", "t_last_dist", "t_last_vafter",
             "t_last_score", "t_last_strong", "t_last_back", "t_dist_since", "ts_last_team", "ts_last_age_s", "ts_run_same", "t_unowned_age_s",
             "t_h_past2", "t_a_past2", "t_h_fut2", "t_a_fut2"]
    G = {k: np.full(T, -1.0) for k in names}
    fr = np.array([d["frame"] for d in touches], int)
    tm = np.array([d["team"] for d in touches], int)
    strong = np.isin(np.array([d["kind"] for d in touches]), ["speed", "both"])
    dist = np.array([d["dist"] if d["dist"] is not None else -1.0 for d in touches])
    vaft = np.array([d["v_after"] if d["v_after"] is not None else -1.0 for d in touches])
    sc = np.array([d["score"] for d in touches])
    back = np.array([d.get("back_frames", 0) for d in touches], float)
    last_xy, _, _, _ = _ffill_bfill(np.asarray(bxy, np.float64), blocks, T)      # most recently seen raw ball position
    w2 = int(round(2 * fps))

    def runs(team):
        run = np.ones(len(team), int)
        for i in range(1, len(team)):
            run[i] = run[i - 1] + 1 if team[i] == team[i - 1] else 1
        return run

    for a, b in blocks.values():
        a, b = int(a), int(b)
        t = np.arange(a, b + 1)
        inb = (fr >= a) & (fr <= b)
        for tag, mask in (("t", tm > 0), ("ts", (tm > 0) & strong), ("u", tm == 0)):
            m = inb & mask
            f = fr[m]
            if not len(f):
                continue
            j = np.searchsorted(f, t, side="right") - 1          # last touch not later than t
            has = j >= 0
            jj = np.clip(j, 0, len(f) - 1)
            age = np.minimum((t - f[jj]) / fps, cap_s)
            if tag == "u":
                G["t_unowned_age_s"][t[has]] = age[has]
                continue
            team = tm[m]
            run = runs(team)
            G[tag + "_last_team"][t[has]] = team[jj[has]]
            G[tag + "_last_age_s"][t[has]] = age[has]
            G[tag + "_run_same"][t[has]] = np.minimum(run[jj], 10)[has]
            if tag == "ts":
                continue
            nxt = j + 1
            hasn = nxt < len(f)
            jn = np.clip(nxt, 0, len(f) - 1)
            G["t_next_team"][t[hasn]] = team[jn[hasn]]
            G["t_next_in_s"][t[hasn]] = np.minimum((f[jn] - t) / fps, cap_s)[hasn]
            both = has & hasn
            G["t_same_team"][t[both]] = (team[jj] == team[jn])[both].astype(float)
            G["t_last_dist"][t[has]] = dist[m][jj][has]
            G["t_last_vafter"][t[has]] = vaft[m][jj][has]
            G["t_last_score"][t[has]] = np.minimum(sc[m][jj], 10)[has]
            G["t_last_strong"][t[has]] = strong[m][jj][has].astype(float)
            G["t_last_back"][t[has]] = back[m][jj][has]
            dd = np.linalg.norm(last_xy[t] - np.asarray(bxy, np.float64)[f][jj], axis=1)
            ok = has & ~np.isnan(dd)
            G["t_dist_since"][t[ok]] = np.minimum(dd[ok], 100)
        n = b - a + 1
        i = np.arange(n)
        for side, code in (("h", 1), ("a", 2)):
            ind = np.zeros(n)
            mm = inb & (tm == code)
            ind[fr[mm] - a] = 1
            cs = np.r_[0, np.cumsum(ind)]
            G["t_%s_past2" % side][a:b + 1] = cs[i + 1] - cs[np.maximum(i - w2, 0)]
            G["t_%s_fut2" % side][a:b + 1] = cs[np.minimum(i + w2 + 1, n)] - cs[i + 1]
    for k in names:
        F["touch_" + k] = G[k]


def kin_features(F, sxy, ref, P, st, vis, fps):
    """Player-ball kinematic features, 11 of them (enabled with --kin_feats). They target short possession spells and boundaries, and describe towards which team's player the ball is heading, how soon it arrives and
    near whom the ball will be 0.5 s later. closing_<team> is the rate of change of the distance to the nearest player (negative means closing in), tti_<team> is the distance divided by the closing speed (0 to 5 s),
    toward_<team> is the cosine between the ball velocity and the direction from the ball to the nearest player, recv_<team> is the smallest distance from the players of that team to the ball position extrapolated 0.5 s ahead; the three differences are h minus a.
    The values are 0 when the ball is not visible or too slow (99 for distances, 5 for tti)."""
    T = len(vis)
    v5 = np.full((T, 2), np.nan)
    v5[5:] = (sxy[5:] - sxy[:-5]) * (fps / 5)
    sp = np.linalg.norm(v5, axis=1)
    ok_v = vis & np.isfinite(sp) & (sp >= 0.5)
    ahead = ref + np.nan_to_num(v5, nan=0.0) * 0.5
    out = {}
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :]
        d = np.linalg.norm(Pt - ref[:, None, :].astype(np.float32), axis=2)
        d = np.where(np.isnan(d), 99.0, d)
        j = np.argmin(d, axis=1)
        d1 = d[np.arange(T), j]
        clos = np.zeros(T)
        clos[5:] = (d1[5:] - d1[:-5]) * (fps / 5)
        ok_c = vis.copy(); ok_c[5:] &= vis[:-5]; ok_c &= d1 < 99
        clos = np.where(ok_c, clos, 0.0)
        tti = np.where(ok_c & (clos < -0.5), np.clip(d1 / np.maximum(-clos, 0.5), 0, 5), 5.0)
        nxy = Pt[np.arange(T), j]
        to_p = nxy - ref
        nrm = np.linalg.norm(to_p, axis=1) * np.where(np.isfinite(sp), sp, 1.0)
        cos = np.where(ok_v & (nrm > 1e-6) & (d1 < 99), np.nan_to_num((to_p * v5).sum(1) / np.maximum(nrm, 1e-6), nan=0.0), 0.0)
        da = np.linalg.norm(Pt - ahead[:, None, :].astype(np.float32), axis=2)
        da = np.where(np.isnan(da), 99.0, da)
        recv = np.where(vis, da.min(1), 99.0)
        out["closing_%s" % side] = clos; out["tti_%s" % side] = tti; out["toward_%s" % side] = cos; out["recv_%s" % side] = recv
    for k in ("closing", "tti", "toward", "recv"):
        F["%s_h" % k] = out["%s_h" % k].astype(np.float32)
        F["%s_a" % k] = out["%s_a" % k].astype(np.float32)
    F["closing_diff"] = F["closing_h"] - F["closing_a"]
    F["toward_diff"] = F["toward_h"] - F["toward_a"]
    F["recv_diff"] = F["recv_h"] - F["recv_a"]


def tta_features(F, ref, P, st, fps, react_s=0.7, vmax=5.0):
    """Time-to-arrival features, 4 of them (enabled with --tta_feats). The arrival time of each player assumes that the player keeps the current velocity during the reaction time react_s
    and then runs straight to the reference ball position at vmax (a simplified time-to-intercept model in the style of Spearman; the parameters 0.7 s and 5 m/s are commonly used values).
    The reference ball position is the current or most recently seen one (ref). When the ball is missing the current positions and velocities of the players are still used, so the features have values inside ball gaps too. Velocities come from
    a 4-frame central difference of the slot coordinates smoothed over 7 frames, capped at 10 m/s; missing frames and block edges count as standing still; identity is not used. tta_h and tta_a are the arrival times of the fastest player of each team (seconds, capped at 25,
    and 25 when the team has nobody on the pitch or there is no reference ball position), tta_diff is h minus a, tta_top3_h is the share of home players among the three shortest arrival times on the pitch (with fewer than three players
    the actual number is used, and -1 when there is no reference ball position)."""
    T = len(ref)
    out, t_all, team_all = {}, [], []
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :].astype(np.float64)
        n = Pt.shape[1]
        sp = _smooth(Pt.reshape(T, -1), 7).reshape(T, n, 2) if n else Pt
        v = np.full_like(sp, np.nan)
        if T > 4:
            v[2:-2] = (sp[4:] - sp[:-4]) * (fps / 4.0)
        spd = np.linalg.norm(v, axis=2)
        v = np.nan_to_num(v * np.where(spd > 10.0, 10.0 / np.maximum(spd, 1e-6), 1.0)[:, :, None], nan=0.0)
        proj = Pt + v * react_s
        d = np.linalg.norm(proj - ref[:, None, :].astype(np.float64), axis=2)
        t = react_s + d / vmax
        t = np.where(np.isnan(t), np.inf, t)
        t_all.append(t); team_all.append(np.full(n, code))
        tm = t.min(1) if n else np.full(T, np.inf)
        out[side] = np.where(np.isfinite(tm), np.minimum(tm, 25.0), 25.0)
    tt = np.concatenate(t_all, axis=1); teams = np.concatenate(team_all)
    order = np.argsort(tt, axis=1, kind="stable")[:, :3]
    top = tt[np.arange(T)[:, None], order]
    valid = np.isfinite(top)
    nvalid = valid.sum(1)
    share = np.where(nvalid > 0, ((teams[order] == 1) & valid).sum(1) / np.maximum(nvalid, 1), -1.0)
    F["tta_h"] = out["h"].astype(np.float32)
    F["tta_a"] = out["a"].astype(np.float32)
    F["tta_diff"] = (out["h"] - out["a"]).astype(np.float32)
    F["tta_top3_h"] = share.astype(np.float32)


def setp_features(F, P, st, hd, L, W, fps, seg, clip=False):
    """Set piece positioning features, 49 columns (enabled with --setp_feats). They only use player positions and need no ball, so they have values when the ball is missing.
    Per team, the smallest distance to any of the four corner flags (corner_d), the signed distance of the player furthest beyond the touchline and beyond the goal line (touch_out, goal_out; negative inside the pitch, close to 0 or positive for a player taking a throw-in or a corner kick),
    the share of players in their own half (own_half; before a kick-off the whole team is in its own half), the number of players in the attacking and in the defending penalty area (box_att, box_def; penalty area size 16.5 m by 40.32 m as in the Laws of the Game),
    the number of players inside the centre circle of radius 9.15 m (centre_n), the distance from the rearmost player (a proxy for the goalkeeper) to the centre of the own goal and the 1 s displacement speed of that player (gk_goal_d, gk_speed),
    the number of players whose 1 s displacement speed is below 0.5 m/s (n_still) and the maximum speed (max_speed). Over the whole pitch, whether the player nearest to the boundary belongs to the home team (edge_home), the signed distance of that player (edge_m),
    and the distance from that player to the nearest other player (edge_iso; a player taking a throw-in often has free space around). Means over the past and future 3 s are added for corner_d, touch_out, goal_out, n_still and max_speed of both teams,
    and for edge_m and edge_iso. Frames without players get fixed values (99 for distances, -99 for signed distances, 0 for counts)."""
    T = len(P)
    keys = []
    corners = np.array([[0.0, 0.0], [0.0, W], [L, 0.0], [L, W]])
    lag = int(round(fps))
    all_xy, all_team = [], []
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :].astype(np.float64)
        n = Pt.shape[1]
        all_xy.append(Pt); all_team.append(np.full(n, code))
        x, y = Pt[:, :, 0], Pt[:, :, 1]
        ok = ~np.isnan(x)
        cnt = np.maximum(ok.sum(1), 1)
        any_ok = ok.any(1)
        with np.errstate(all="ignore"):
            dc = np.min(np.sqrt((x[:, :, None] - corners[:, 0]) ** 2 + (y[:, :, None] - corners[:, 1]) ** 2), axis=2)
            F["corner_d_%s" % side] = np.where(any_ok, np.nan_to_num(np.nanmin(np.where(ok, dc, np.inf), 1), posinf=99.0), 99.0).astype(np.float32)
            tout = np.maximum(-y, y - W)
            gout = np.maximum(-x, x - L)
            F["touch_out_%s" % side] = np.where(any_ok, np.nanmax(np.where(ok, tout, -np.inf), 1), -99.0).astype(np.float32)
            F["goal_out_%s" % side] = np.where(any_ok, np.nanmax(np.where(ok, gout, -np.inf), 1), -99.0).astype(np.float32)
            xd = np.where(hd[:, None] > 0, x, L - x)              # coordinate along the home attacking direction, the home team attacks towards xd=L and the away team towards xd=0
            own = (xd < L / 2) if code == 1 else (xd > L / 2)
            F["own_half_%s" % side] = ((own & ok).sum(1) / cnt).astype(np.float32)
            in_w = np.abs(y - W / 2) < 20.16
            box_hi = (xd > L - 16.5) & in_w & ok                  # penalty area attacked by the home team
            box_lo = (xd < 16.5) & in_w & ok                      # penalty area attacked by the away team
            F["box_att_%s" % side] = (box_hi if code == 1 else box_lo).sum(1).astype(np.float32)
            F["box_def_%s" % side] = (box_lo if code == 1 else box_hi).sum(1).astype(np.float32)
            F["centre_n_%s" % side] = ((np.hypot(x - L / 2, y - W / 2) < 9.15) & ok).sum(1).astype(np.float32)
            sp = _smooth(Pt.reshape(T, -1), 7).reshape(T, n, 2) if n else Pt
            d1 = np.full((T, n), np.nan)
            if T > lag:
                d1[lag:] = np.linalg.norm(sp[lag:] - sp[:-lag], axis=2) * fps / lag
            rear = np.where(ok, xd, np.inf if code == 1 else -np.inf)
            ir = rear.argmin(1) if code == 1 else rear.argmax(1)   # rearmost player; the home team attacks towards xd=L so take the smallest xd, and the largest for the away team
            ti = np.arange(T)
            gx = 0.0 if code == 1 else L                           # position of the own goal in xd coordinates
            F["gk_goal_d_%s" % side] = np.where(any_ok, np.nan_to_num(np.hypot(xd[ti, ir] - gx, y[ti, ir] - W / 2), nan=99.0), 99.0).astype(np.float32)
            F["gk_speed_%s" % side] = np.nan_to_num(d1[ti, ir], nan=-1.0).astype(np.float32)
            F["n_still_%s" % side] = np.nansum(d1 < 0.5, axis=1).astype(np.float32)
            has_d1 = (~np.isnan(d1)).any(1)
            F["max_speed_%s" % side] = np.where(has_d1, np.nanmax(np.where(np.isnan(d1), -np.inf, d1), 1), -1.0).astype(np.float32)
        keys += ["corner_d_%s" % side, "touch_out_%s" % side, "goal_out_%s" % side, "n_still_%s" % side, "max_speed_%s" % side]
    Pa = np.concatenate(all_xy, axis=1); teams = np.concatenate(all_team)
    with np.errstate(all="ignore"):
        m = np.maximum.reduce([-Pa[:, :, 1], Pa[:, :, 1] - W, -Pa[:, :, 0], Pa[:, :, 0] - L])
        okm = ~np.isnan(m)
        anyp = okm.any(1)
        ie = np.where(okm, m, -np.inf).argmax(1)
        ti = np.arange(T)
        F["edge_home"] = np.where(anyp, (teams[ie] == 1).astype(np.float32), -1.0).astype(np.float32)
        F["edge_m"] = np.where(anyp, m[ti, ie], -99.0).astype(np.float32)
        dd = np.linalg.norm(Pa - Pa[ti, ie][:, None, :], axis=2)
        dd[ti, ie] = np.nan
        ddf = (~np.isnan(dd)).any(1)
        F["edge_iso"] = np.where(ddf, np.nanmin(np.where(np.isnan(dd), np.inf, dd), 1), 99.0).astype(np.float32)
    keys += ["edge_m", "edge_iso"]
    if clip:                                                  # clip to the physical range, set only from the pitch size and the upper limit of human speed, not tuned on data
        for sd_ in ("h", "a"):
            F["corner_d_" + sd_] = np.clip(F["corner_d_" + sd_], 0.0, 60.0)
            F["touch_out_" + sd_] = np.clip(F["touch_out_" + sd_], -34.0, 5.0)
            F["goal_out_" + sd_] = np.clip(F["goal_out_" + sd_], -52.5, 5.0)
            F["gk_goal_d_" + sd_] = np.clip(F["gk_goal_d_" + sd_], 0.0, 60.0)
            F["gk_speed_" + sd_] = np.clip(F["gk_speed_" + sd_], -1.0, 12.0)
            F["max_speed_" + sd_] = np.clip(F["max_speed_" + sd_], -1.0, 12.0)
        F["edge_m"] = np.clip(F["edge_m"], -10.0, 5.0)
        F["edge_iso"] = np.clip(F["edge_iso"], 0.0, 30.0)
    w = int(round(3 * fps))
    for k in keys:
        xv = np.asarray(F[k], np.float64)
        F["%s_past3" % k] = window_mean(xv, w, seg).astype(np.float32)
        F["%s_fut3" % k] = window_mean(xv, w, seg, True).astype(np.float32)


def shape_features(F, P, st, hd, L, W, fps, seg):
    """Team shape behaviour features, 53 columns (enabled with --shape_feats). They only use players and need no ball. Per team, the depth (foremost player minus second rearmost player, shape_len),
    the width (shape_wid), the height of the defensive line (coordinate of the second rearmost player along the attacking direction of the team, shape_line), the mean longitudinal position without the rearmost player (shape_mid), the mean 1 s displacement velocity along the attacking direction of the team (v_fwd),
    the share of players running forwards and backwards faster than 1 m/s (f_fwd, f_back) and the mean distance from each player to the nearest opponent (mark). Over the whole pitch, the mean velocity of both teams along the home attacking direction (v_all; above 0 means both teams as a whole are moving towards the away goal).
    Means over the past and future 1 s and 3 s are added for v_all, f_fwd, f_back, mark and shape_len. Everything is clipped to its physical range, and frames with missing players get -1 (0 for v_fwd and v_all)."""
    T = len(P)
    lag = int(round(fps))
    H_ = {}
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :].astype(np.float64)
        n = Pt.shape[1]
        x, y = Pt[:, :, 0], Pt[:, :, 1]
        ok = ~np.isnan(x)
        cnt = ok.sum(1)
        xd = np.where(hd[:, None] > 0, x, L - x)
        att = xd if code == 1 else L - xd                     # coordinate along the attacking direction of the team, larger means closer to the goal being attacked
        s_ = np.sort(np.where(ok, att, np.inf), axis=1)
        deep1, deep2 = s_[:, 0], (s_[:, 1] if n > 1 else s_[:, 0])
        front = np.where(cnt > 0, np.max(np.where(ok, att, -np.inf), axis=1), -1.0)
        F["shape_len_%s" % side] = np.where(cnt >= 2, np.clip(front - deep2, 0.0, 105.0), -1.0).astype(np.float32)
        ymax = np.max(np.where(ok, y, -np.inf), axis=1); ymin = np.min(np.where(ok, y, np.inf), axis=1)
        F["shape_wid_%s" % side] = np.where(cnt >= 2, np.clip(ymax - ymin, 0.0, 68.0), -1.0).astype(np.float32)
        F["shape_line_%s" % side] = np.where(cnt >= 2, np.clip(deep2, 0.0, 105.0), -1.0).astype(np.float32)
        sum_att = np.sum(np.where(ok, att, 0.0), axis=1)
        F["shape_mid_%s" % side] = np.where(cnt >= 2, np.clip((sum_att - np.where(cnt > 0, deep1, 0.0)) / np.maximum(cnt - 1, 1), 0.0, 105.0), -1.0).astype(np.float32)
        sp = _smooth(Pt.reshape(T, -1), 7).reshape(T, n, 2) if n else Pt
        sx = np.where(hd[:, None] > 0, sp[:, :, 0], L - sp[:, :, 0])
        satt = sx if code == 1 else L - sx
        v = np.full((T, n), np.nan)
        if T > lag:
            v[lag:] = (satt[lag:] - satt[:-lag]) * fps / lag
        v = np.where(np.abs(v) > 12.0, np.nan, v)              # jumps caused by the switch between halves and by dropouts count as missing
        vok = ~np.isnan(v)
        nv = vok.sum(1)
        F["v_fwd_%s" % side] = np.where(nv > 0, np.clip(np.sum(np.where(vok, v, 0.0), axis=1) / np.maximum(nv, 1), -8.0, 8.0), 0.0).astype(np.float32)
        F["f_fwd_%s" % side] = np.where(nv > 0, ((v > 1.0) & vok).sum(1) / np.maximum(nv, 1), -1.0).astype(np.float32)
        F["f_back_%s" % side] = np.where(nv > 0, ((v < -1.0) & vok).sum(1) / np.maximum(nv, 1), -1.0).astype(np.float32)
        H_[code] = Pt.astype(np.float32)
    for side, a_, b_ in (("h", 1, 2), ("a", 2, 1)):
        Pa, Pb = H_[a_], H_[b_]
        d = np.sqrt(((Pa[:, :, None, :] - Pb[:, None, :, :]) ** 2).sum(3))
        d = np.where(np.isnan(d), np.inf, d)
        dm = d.min(2)
        okm = np.isfinite(dm)
        nm = okm.sum(1)
        F["mark_%s" % side] = np.where(nm > 0, np.clip(np.sum(np.where(okm, dm, 0.0), axis=1) / np.maximum(nm, 1), 0.0, 30.0), -1.0).astype(np.float32)
    F["v_all"] = ((F["v_fwd_h"] - F["v_fwd_a"]) / 2.0).astype(np.float32)
    keys = ["v_all"] + ["%s_%s" % (k_, side) for side in ("h", "a") for k_ in ("f_fwd", "f_back", "mark", "shape_len")]
    for k in keys:
        xv = np.asarray(F[k], np.float64)
        for w_s in (1, 3):
            w = int(round(w_s * fps))
            F["%s_past%d" % (k, w_s)] = window_mean(xv, w, seg).astype(np.float32)
            F["%s_fut%d" % (k, w_s)] = window_mean(xv, w, seg, True).astype(np.float32)


def _player_motion(Pt, fps):
    """Per-player 1 s displacement velocity in world coordinates (T, n, 2), in m/s. Smoothed over 7 frames first; values above 12 m/s count as missing (jumps caused by the switch between halves and by dropouts)"""
    T, n = Pt.shape[:2]
    lag = int(round(fps))
    v = np.full((T, n, 2), np.nan)
    if n and T > lag:
        sp = _smooth(Pt.reshape(T, -1), 7).reshape(T, n, 2)
        v[lag:] = (sp[lag:] - sp[:-lag]) * fps / lag
        v[np.linalg.norm(v, axis=2) > 12.0] = np.nan
    return v


def press_features(F, P, st, hd, L, W, fps, seg, ref, ref_next):
    """Pressing focus and positioning features, 52 columns (enabled with --press_feats). Computed for each team X.
    press_max, the number of opponents within 5 m of the player of X who is most tightly surrounded; conv_max, for the player of X approached by the most opponents, the number of opponents within 15 m running towards that player faster than 2 m/s;
    focal_speed, focal_att, focal_ref, the speed of the focal player (the player with the largest sum of the two counts above; frames where the sum is 0 have no focal player and get -1), the coordinate of that player along the attacking direction of the team and the distance to the reference ball position;
    free, the number of players with no opponent within 5 m; off_gap, the foremost player of X minus the defensive line of the opponent (the second rearmost opponent), positive when beyond it; n_line, the number of players of X within 5 m of the defensive line of the opponent or beyond it;
    n_gs, n_gs_next, the number of players of X between their own goal and the ball, relative to the current or most recently seen ball position and to the current or next seen ball position, -1 when the ball position is unknown.
    Means over the past and future 1 s and 3 s are added for press_max, conv_max, free and n_gs. The radii and speed thresholds are generic scales fixed in advance and are not tuned on the competition data."""
    P_, V_, A_ = {}, {}, {}
    for code in (1, 2):
        Pt = P[:, st == code, :].astype(np.float32)
        P_[code] = Pt
        V_[code] = _player_motion(Pt, fps).astype(np.float32)
        xd = np.where(hd[:, None] > 0, Pt[:, :, 0], L - Pt[:, :, 0])
        A_[code] = xd if code == 1 else L - xd
    for side, a_, b_ in (("h", 1, 2), ("a", 2, 1)):
        Pa, Pb, Vb, Aa, Ab = P_[a_], P_[b_], V_[b_], A_[a_], A_[b_]
        okA = ~np.isnan(Pa[:, :, 0])
        okB = ~np.isnan(Pb[:, :, 0])
        cntA, cntB = okA.sum(1), okB.sum(1)
        rel = Pa[:, :, None, :] - Pb[:, None, :, :]              # vector from the opponent to the player
        d = np.sqrt((rel ** 2).sum(3))
        with np.errstate(all="ignore"):
            close = (Vb[:, None, :, 0] * rel[:, :, :, 0] + Vb[:, None, :, 1] * rel[:, :, :, 1]) / d
        n5 = (d < 5.0).sum(2)
        conv = ((d < 15.0) & (close > 2.0)).sum(2)
        F["press_max_%s" % side] = np.where(cntA > 0, np.max(np.where(okA, n5, 0), axis=1), -1).astype(np.float32)
        F["conv_max_%s" % side] = np.where(cntA > 0, np.max(np.where(okA, conv, 0), axis=1), -1).astype(np.float32)
        score = np.where(okA, n5 + conv, -1)
        i_ = score.argmax(1)
        has = score[np.arange(len(P)), i_] > 0
        idx = np.arange(len(P))
        fx = Pa[idx, i_] if Pa.shape[1] else np.full((len(P), 2), np.nan)
        fv = _player_motion(Pa, fps)[idx, i_] if Pa.shape[1] else np.full((len(P), 2), np.nan)
        spd = np.sqrt((fv ** 2).sum(1))
        F["focal_speed_%s" % side] = np.where(has & ~np.isnan(spd), np.clip(spd, 0.0, 12.0), -1.0).astype(np.float32)
        F["focal_att_%s" % side] = np.where(has, np.clip(Aa[idx, i_], 0.0, 105.0), -1.0).astype(np.float32)
        dref = np.sqrt(((fx - ref) ** 2).sum(1))
        F["focal_ref_%s" % side] = np.where(has & ~np.isnan(dref), np.clip(dref, 0.0, 60.0), -1.0).astype(np.float32)
        dm = np.min(np.where(np.isnan(d), np.inf, d), axis=2)
        F["free_%s" % side] = np.where(cntA > 0, ((dm > 5.0) & np.isfinite(dm) & okA).sum(1), -1).astype(np.float32)
        s_ = np.sort(np.where(okB, Ab, np.inf), axis=1)
        line_b = s_[:, 1] if s_.shape[1] > 1 else s_[:, 0]       # second rearmost opponent, in the attacking coordinate of the opponent
        line_in_a = L - line_b                                    # converted to the attacking coordinate of X
        front = np.max(np.where(okA, Aa, -np.inf), axis=1)
        both = (cntA > 0) & (cntB >= 2)
        F["off_gap_%s" % side] = np.where(both, np.clip(front - line_in_a, -60.0, 30.0), 0.0).astype(np.float32)
        with np.errstate(all="ignore"):
            F["n_line_%s" % side] = np.where(both, ((Aa >= line_in_a[:, None] - 5.0) & okA).sum(1), -1).astype(np.float32)
        for tag, R in (("", ref), ("_next", ref_next)):
            bd = np.where(hd > 0, R[:, 0], L - R[:, 0])
            batt = bd if a_ == 1 else L - bd
            with np.errstate(all="ignore"):
                ngs = ((Aa < batt[:, None]) & okA).sum(1)
            F["n_gs%s_%s" % (tag, side)] = np.where(np.isnan(batt) | (cntA == 0), -1, ngs).astype(np.float32)
    keys = ["%s_%s" % (k_, side) for side in ("h", "a") for k_ in ("press_max", "conv_max", "free", "n_gs")]
    for k in keys:
        xv = np.asarray(F[k], np.float64)
        for w_s in (1, 3):
            w = int(round(w_s * fps))
            F["%s_past%d" % (k, w_s)] = window_mean(xv, w, seg).astype(np.float32)
            F["%s_fut%d" % (k, w_s)] = window_mean(xv, w, seg, True).astype(np.float32)


def roster_features(F, P, st, hd, L, W, fps):
    """Per-player features sorted by position, 110 columns (enabled with --roster_feats). For each team the players on the pitch are sorted from back to front by their coordinate along the attacking direction of the team, and the first 11 ranks are kept.
    Each rank gets 5 columns, the coordinate along the attacking direction of the team (att), the lateral coordinate (y), the velocity along the attacking direction (vf), the lateral velocity (vl) and the distance to the nearest opponent (dn). Player identity is not used; the ordering only uses the positions in the current frame.
    For missing ranks att, y and dn are -1 and vf and vl are 0. This lets the network learn from the positions of individual players, not only from aggregates."""
    T = len(P)
    Pts = {c: P[:, st == c, :].astype(np.float32) for c in (1, 2)}
    for side, a_, b_ in (("h", 1, 2), ("a", 2, 1)):
        Pa, Pb = Pts[a_], Pts[b_]
        n = Pa.shape[1]
        xd = np.where(hd[:, None] > 0, Pa[:, :, 0], L - Pa[:, :, 0])
        att = xd if a_ == 1 else L - xd
        ok = ~np.isnan(att)
        v = _player_motion(Pa, fps)
        sgn = np.where(hd > 0, 1.0, -1.0) * (1.0 if a_ == 1 else -1.0)
        vf = v[:, :, 0] * sgn[:, None]
        rel = Pa[:, :, None, :] - Pb[:, None, :, :]
        d = np.sqrt((rel ** 2).sum(3))
        dn = np.min(np.where(np.isnan(d), np.inf, d), axis=2)
        order = np.argsort(np.where(ok, att, np.inf), axis=1)
        for r in range(11):
            if r < n:
                i_ = order[:, r]
                idx = np.arange(T)
                has = ok[idx, i_]
                F["att_%s_r%d" % (side, r)] = np.where(has, np.clip(att[idx, i_], 0.0, 105.0), -1.0).astype(np.float32)
                F["y_%s_r%d" % (side, r)] = np.where(has, np.clip(Pa[idx, i_, 1], 0.0, W), -1.0).astype(np.float32)
                F["vf_%s_r%d" % (side, r)] = np.nan_to_num(np.clip(vf[idx, i_], -12.0, 12.0) * has, nan=0.0).astype(np.float32)
                F["vl_%s_r%d" % (side, r)] = np.nan_to_num(np.clip(v[idx, i_, 1], -12.0, 12.0) * has, nan=0.0).astype(np.float32)
                dd = dn[idx, i_]
                F["dn_%s_r%d" % (side, r)] = np.where(has & np.isfinite(dd), np.clip(dd, 0.0, 30.0), -1.0).astype(np.float32)
            else:
                F["att_%s_r%d" % (side, r)] = np.full(T, -1.0, np.float32)
                F["y_%s_r%d" % (side, r)] = np.full(T, -1.0, np.float32)
                F["vf_%s_r%d" % (side, r)] = np.zeros(T, np.float32)
                F["vl_%s_r%d" % (side, r)] = np.zeros(T, np.float32)
                F["dn_%s_r%d" % (side, r)] = np.full(T, -1.0, np.float32)


def dir_features(F, P, st, hd, L, W, fps, seg):
    """Direction of movement features, 32 columns (enabled with --dir_feats). They only use players. Per team 8 columns, the mean velocity along the attacking direction of the team over displacement lags of 0.5, 2 and 4 s (vf05, vf2, vf4),
    the mean lateral velocity (vlat), the mean 1 s velocity of three groups by longitudinal rank in the current frame (vback, the three rearmost players after leaving out the last one; vmid, the next three; vfront, the three foremost) and the share of players running forwards minus the share running backwards (fnet, threshold 1 m/s).
    Over the whole pitch 6 columns plus 8 window columns, the combined velocity of both teams along the home attacking direction at the three lags (va05, va2, va4), the sum of the velocities of both teams, each along its own attacking direction (vsum; above 0 means they move towards each other, below 0 means both retreat), the mean of the combined velocity over the next 1 s and 3 s minus the mean over the same length in the past (dva1, dva3; largest when the direction flips),
    the seconds since the last and until the next sign flip of the combined velocity (va_flip_past, va_flip_fut; a direction only counts when the absolute value exceeds 0.5 m/s, capped at 30 s), the means of the combined velocity over the past and future 5 s and 10 s, and the means of fnet over the past and future 5 s.
    Velocities are clipped to their physical range; displacements across halves or above 12 m/s count as missing. The thresholds are generic scales fixed in advance and are not tuned on the competition data."""
    T = len(P)
    arange = np.arange(T)
    V1 = {}
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :].astype(np.float64)
        n = Pt.shape[1]
        sp = _smooth(Pt.reshape(T, -1), 7).reshape(T, n, 2) if n else Pt
        sgn = np.where(hd > 0, 1.0, -1.0) * (1.0 if code == 1 else -1.0)
        xd = np.where(hd[:, None] > 0, Pt[:, :, 0], L - Pt[:, :, 0])
        att = xd if code == 1 else L - xd
        ok = ~np.isnan(att)

        def disp_v(k, lag_s):
            lag = max(1, int(round(lag_s * fps)))
            v = np.full((T, n), np.nan)
            if n and T > lag:
                v[lag:] = (sp[lag:, :, k] - sp[:-lag, :, k]) * fps / lag
                v[lag:][seg[lag:] != seg[:-lag]] = np.nan
            return np.where(np.abs(v) > 12.0, np.nan, v)

        def nmean(v, mask=None):
            okv = ~np.isnan(v) if mask is None else (~np.isnan(v) & mask)
            c = okv.sum(1)
            return np.where(c > 0, np.sum(np.where(okv, v, 0.0), axis=1) / np.maximum(c, 1), 0.0)

        for tag, lag_s in (("05", 0.5), ("2", 2.0), ("4", 4.0)):
            F["vf%s_%s" % (tag, side)] = np.clip(nmean(disp_v(0, lag_s) * sgn[:, None]), -8.0, 8.0).astype(np.float32)
        F["vlat_%s" % side] = np.clip(nmean(disp_v(1, 1.0)), -8.0, 8.0).astype(np.float32)
        v1 = disp_v(0, 1.0) * sgn[:, None]
        V1[side] = nmean(v1)
        order = np.argsort(np.where(ok, att, np.inf), axis=1)
        vs = np.take_along_axis(v1, order, axis=1)
        cnt = ok.sum(1)[:, None]
        rk = np.arange(n)[None, :]
        F["vback_%s" % side] = np.clip(nmean(vs, (rk >= 1) & (rk <= 3) & (rk < cnt)), -8.0, 8.0).astype(np.float32)
        F["vmid_%s" % side] = np.clip(nmean(vs, (rk >= 4) & (rk <= 6) & (rk < cnt)), -8.0, 8.0).astype(np.float32)
        F["vfront_%s" % side] = np.clip(nmean(vs, (rk >= cnt - 3) & (rk < cnt)), -8.0, 8.0).astype(np.float32)
        okv = ~np.isnan(v1)
        nv = okv.sum(1)
        F["fnet_%s" % side] = np.where(nv > 0, (((v1 > 1.0) & okv).sum(1) - ((v1 < -1.0) & okv).sum(1)) / np.maximum(nv, 1), 0.0).astype(np.float32)
    va = (V1["h"] - V1["a"]) / 2.0
    for tag in ("05", "2", "4"):
        F["va%s" % tag] = ((F["vf%s_h" % tag] - F["vf%s_a" % tag]) / 2.0).astype(np.float32)
    F["vsum"] = np.clip(V1["h"] + V1["a"], -16.0, 16.0).astype(np.float32)
    for w_s, nm in ((1, "dva1"), (3, "dva3")):
        w = int(round(w_s * fps))
        F[nm] = (window_mean(va, w, seg, True) - window_mean(va, w, seg)).astype(np.float32)
    sgn_ = np.where(va > 0.5, 1, np.where(va < -0.5, -1, 0))
    idx = np.maximum.accumulate(np.where(sgn_ != 0, arange, 0))
    state = sgn_[idx]
    flip = np.zeros(T, bool)
    flip[1:] = (state[1:] != state[:-1]) & (state[:-1] != 0) & (seg[1:] == seg[:-1])
    cap = 30.0
    last = np.maximum.accumulate(np.where(flip, arange, -1))
    F["va_flip_past"] = np.where(last >= 0, np.minimum((arange - last) / fps, cap), cap).astype(np.float32)
    nxt = np.minimum.accumulate(np.where(flip, arange, T)[::-1])[::-1]
    F["va_flip_fut"] = np.where(nxt < T, np.minimum((nxt - arange) / fps, cap), cap).astype(np.float32)
    for w_s in (5, 10):
        w = int(round(w_s * fps))
        F["va_past%d" % w_s] = window_mean(va, w, seg).astype(np.float32)
        F["va_fut%d" % w_s] = window_mean(va, w, seg, True).astype(np.float32)
    for side in ("h", "a"):
        xv = np.asarray(F["fnet_%s" % side], np.float64)
        w = int(round(5 * fps))
        F["fnet_%s_past5" % side] = window_mean(xv, w, seg).astype(np.float32)
        F["fnet_%s_fut5" % side] = window_mean(xv, w, seg, True).astype(np.float32)


def setp_frame_arrays(M):
    """Per-frame columns of setp_features for the segment scorer; returns {column name: (T,) array}. The attacking direction is determined as in build_features"""
    dirs = _attack_dir(M, M["blocks"])
    seg = M["period"]
    hd = np.array([dirs.get(int(x), 1) for x in seg])
    L, W = M["pitch"]
    F = {}
    setp_features(F, M["player_xy"].astype(np.float32), M["slot_team"], hd, L, W, M["fps"], seg)
    return F


def shape_frame_arrays(M):
    """Per-frame columns of shape_features for the segment scorer; returns {column name: (T,) array}. The attacking direction is determined as in build_features"""
    dirs = _attack_dir(M, M["blocks"])
    seg = M["period"]
    hd = np.array([dirs.get(int(x), 1) for x in seg])
    L, W = M["pitch"]
    F = {}
    shape_features(F, M["player_xy"].astype(np.float32), M["slot_team"], hd, L, W, M["fps"], seg)
    return F


def comove_features(F, sxy, P, st, vis, fps, seg, v2=False):
    """Player-ball co-movement features, 9 of them (enabled with --comove_feats). Under tight marking the ball carrier and the ball move together over a few tenths of a second,
    while the marking defender cannot keep up with the ball. In each frame, for every player within 3 m of the ball, the difference between the 0.2 s displacement velocity of the player and the ball velocity (m/s) is computed. Its minimum per team gives comove_dv_h and comove_dv_a
    (10 when there is no such player or the ball is not visible). Added to these are the difference between the two teams comove_dv_diff, the team with the best co-movement comove_team (home 1, away -1, none 0),
    the mean of comove_team over the past and future 0.5 s and 1 s (how long it lasted), and comove_contest, an indicator that both teams have a player within 1.5 m of the ball."""
    T = len(vis)
    k = max(1, int(round(0.2 * fps)))
    vb = np.full((T, 2), np.nan)
    vb[k:] = (sxy[k:] - sxy[:-k]) * (fps / k)
    Pf = P.astype(np.float64)
    vp = np.full(Pf.shape, np.nan)
    vp[k:] = (Pf[k:] - Pf[:-k]) * (fps / k)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(Pf - sxy[:, None, :], axis=2)
        dv = np.linalg.norm(vp - vb[:, None, :], axis=2)
    ok = np.asarray(vis, bool)[:, None] & np.isfinite(d) & np.isfinite(dv)
    near3 = ok & (d < 3.0)
    score = np.where(near3, dv, np.inf)
    out = {}
    for side, code in (("h", 1), ("a", 2)):
        m = np.min(np.where((st == code)[None, :], score, np.inf), axis=1)
        out[side] = np.where(np.isfinite(m), np.minimum(m, 10.0), 10.0)
        F["comove_dv_%s" % side] = out[side]
    F["comove_dv_diff"] = out["h"] - out["a"]
    anyp = (out["h"] < 10) | (out["a"] < 10)
    ind = np.where(anyp, np.where(out["h"] < out["a"], 1.0, -1.0), 0.0)
    F["comove_team"] = ind
    for w_s in (0.5, 1.0):
        w = max(1, int(round(w_s * fps)))
        F["comove_team_past%gs" % w_s] = window_mean(ind, w, seg)
        F["comove_team_fut%gs" % w_s] = window_mean(ind, w, seg, True)
    dd = np.where(ok, d, np.inf)
    F["comove_contest"] = ((np.min(np.where((st == 1)[None, :], dd, np.inf), axis=1) < 1.5) &
                           (np.min(np.where((st == 2)[None, :], dd, np.inf), axis=1) < 1.5)).astype(np.float32)
    if not v2:
        return
    # with v2 (--comove_v2) the 9 columns above are unchanged and 9 more are added, with one more time scale and one more radius, plus the direction of movement
    k4 = max(1, int(round(0.4 * fps)))
    vb4 = np.full((T, 2), np.nan)
    vb4[k4:] = (sxy[k4:] - sxy[:-k4]) * (fps / k4)
    vp4 = np.full(Pf.shape, np.nan)
    vp4[k4:] = (Pf[k4:] - Pf[:-k4]) * (fps / k4)
    with np.errstate(invalid="ignore"):
        dv4 = np.linalg.norm(vp4 - vb4[:, None, :], axis=2)
    ok4 = np.asarray(vis, bool)[:, None] & np.isfinite(d) & np.isfinite(dv4) & (d < 3.0)
    score4 = np.where(ok4, dv4, np.inf)
    o4 = {}
    for side, code in (("h", 1), ("a", 2)):
        m = np.min(np.where((st == code)[None, :], score4, np.inf), axis=1)
        o4[side] = np.where(np.isfinite(m), np.minimum(m, 10.0), 10.0)
        F["comove4_dv_%s" % side] = o4[side]
    F["comove4_dv_diff"] = o4["h"] - o4["a"]
    any4 = (o4["h"] < 10) | (o4["a"] < 10)
    F["comove4_team"] = np.where(any4, np.where(o4["h"] < o4["a"], 1.0, -1.0), 0.0)
    w2 = max(1, int(round(2.0 * fps)))
    F["comove_team_past2s"] = window_mean(ind, w2, seg)
    F["comove_team_fut2s"] = window_mean(ind, w2, seg, True)
    sb = np.linalg.norm(vb, axis=1)
    mv = np.isfinite(sb) & (sb > 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosv = (vp * vb[:, None, :]).sum(axis=2) / (np.linalg.norm(vp, axis=2) * sb[:, None])
    okc = near3 & mv[:, None] & np.isfinite(cosv)
    for side, code in (("h", 1), ("a", 2)):
        m = np.max(np.where((st == code)[None, :] & okc, cosv, -np.inf), axis=1)
        F["comove_cos_%s" % side] = np.where(np.isfinite(m), m, 0.0)
    near5 = ok & (d < 5.0)
    score5 = np.where(near5, dv, np.inf)
    o5 = []
    for code in (1, 2):
        m = np.min(np.where((st == code)[None, :], score5, np.inf), axis=1)
        o5.append(np.where(np.isfinite(m), np.minimum(m, 10.0), 10.0))
    F["comove5_dv_diff"] = o5[0] - o5[1]


PSET_F = 11   # number of columns per player


def pset_features(F, ref, sxy, P, st, hd, vis, fps, seg, K):
    """Ball-centred per-player features, K players times PSET_F columns (enabled with pset_feats=K). In each frame the K players nearest to the reference ball position ref (the current one when the ball is visible, otherwise the most recently seen one)
    are taken regardless of team and sorted from nearest to farthest. The columns per player are team (home 1, away -1, missing 0), dist the distance to the ball divided by 10, dxown the offset ahead of the ball (positive) or behind it along the attacking direction of the player's own team divided by 10,
    spd the 0.2 s displacement speed of the player divided by 5, dv the velocity difference to the ball divided by 5 (2 when it cannot be computed), cos the cosine with the direction of movement of the ball (0 when the ball speed is below 1 m/s or the player speed below 0.5),
    close the rate of change of the distance to the ball over 0.2 s divided by 5 (negative means closing in), dm04 dp04 dm1 dp1 the distance from the same slot to the ball 0.4 s before and after and 1 s before and after, divided by 10 (the current distance is used across halves).
    When home and away are swapped only team changes sign and everything else stays the same. The columns are named pset<rank>_<name>, ordered by rank first and then by column, and placed at the end of the feature table; seq_model --pset_pool splits them in this order"""
    T, S = P.shape[0], P.shape[1]
    Pf = P.astype(np.float64)
    with np.errstate(invalid="ignore"):
        D = np.linalg.norm(Pf - ref[:, None, :], axis=2)                 # (T, S), NaN for a missing player or when there is no reference ball position
    Ds = np.where(np.isfinite(D), D, np.inf)
    idx = np.argsort(Ds, axis=1, kind="stable")[:, :K]                   # (T, K)
    if idx.shape[1] < K:
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], K - idx.shape[1], axis=1)], axis=1)
    take = lambda A: np.take_along_axis(A, idx, axis=1)
    d = take(Ds)
    okp = np.isfinite(d)
    k = max(1, int(round(0.2 * fps)))
    vb = np.full((T, 2), np.nan)
    vb[k:] = (sxy[k:] - sxy[:-k]) * (fps / k)
    vp = np.full(Pf.shape, np.nan)
    vp[k:] = (Pf[k:] - Pf[:-k]) * (fps / k)
    seg = np.asarray(seg)
    same = np.zeros(T, bool); same[k:] = seg[k:] == seg[:-k]
    vb[~same] = np.nan; vp[~same] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        spd = np.linalg.norm(vp, axis=2)
        dv = np.linalg.norm(vp - vb[:, None, :], axis=2)
        sb = np.linalg.norm(vb, axis=1)
        cosv = (vp * vb[:, None, :]).sum(axis=2) / (spd * sb[:, None])
        cosv = np.where((sb[:, None] > 1.0) & (spd > 0.5) & np.isfinite(cosv), cosv, 0.0)
        close = np.full(D.shape, np.nan)
        close[k:] = (D[k:] - D[:-k]) * (fps / k)
        close[~same] = np.nan
    vis_ = np.asarray(vis, bool)[:, None]

    def shifted(sh):                                                    # positive sh looks into the past, negative into the future; distance from the same slot at that moment to the reference ball position at that moment
        out = np.full(D.shape, np.nan)
        n = abs(sh)
        if n < T:
            if sh > 0:
                out[n:] = D[:-n]; bad = np.ones(T, bool); bad[n:] = seg[n:] != seg[:-n]
            else:
                out[:-n] = D[n:]; bad = np.ones(T, bool); bad[:-n] = seg[:-n] != seg[n:]
            out[bad] = np.nan
        return out
    team = np.where(st == 1, 1.0, np.where(st == 2, -1.0, 0.0))        # (S,)
    dirt = team[None, :] * np.asarray(hd, np.float64)[:, None]         # attacking direction of the player's own team, hd for the home team and -hd for the away team
    dxown = (Pf[:, :, 0] - ref[:, 0][:, None]) * dirt
    k4, k10 = max(1, int(round(0.4 * fps))), max(1, int(round(1.0 * fps)))
    cols = {
        "team": np.where(okp, team[idx], 0.0),
        "dist": np.where(okp, np.minimum(d, 30.0), 30.0) / 10.0,
        "dxown": np.where(okp, np.clip(np.nan_to_num(take(dxown)), -30.0, 30.0), 0.0) / 10.0,
        "spd": np.where(okp, np.minimum(np.nan_to_num(take(spd)), 12.0), 0.0) / 5.0,
        "dv": np.where(okp & take(np.isfinite(dv) & vis_), np.minimum(np.nan_to_num(take(dv), nan=10.0), 10.0), 10.0) / 5.0,
        "cos": np.where(okp & take(np.broadcast_to(vis_, D.shape)), take(cosv), 0.0),
        "close": np.where(okp, np.clip(np.nan_to_num(take(close)), -10.0, 10.0), 0.0) / 5.0,
    }
    for nm, sh in (("dm04", k4), ("dp04", -k4), ("dm1", k10), ("dp1", -k10)):
        x = take(shifted(sh))
        cols[nm] = np.where(okp, np.minimum(np.where(np.isfinite(x), x, d), 30.0), 30.0) / 10.0
    assert len(cols) == PSET_F
    for r in range(K):
        for nm, v in cols.items():
            F["pset%d_%s" % (r, nm)] = v[:, r].astype(np.float32)


def build_features(M, smooth_w=7, gap_feats=False, touch_feats=False, touch_set="robust", kin_feats=False, tta_feats=False, setp_feats=False, shape_feats=False, press_feats=False, roster_feats=False, dir_feats=False, comove_feats=False, pset_feats=0):
    fps = M["fps"]
    L, W = M["pitch"]
    T = len(M["t_ms"])
    blocks = M["blocks"]
    seg = M["period"]
    bxy, vis = M["ball_xy"], M["ball_visible"]
    P = M["player_xy"].astype(np.float32)
    st = M["slot_team"]
    F = {}
    sxy = _smooth(bxy, smooth_w)
    last, age_l, nxt, age_n = _ffill_bfill(sxy, blocks, T)
    ref = np.where(np.isnan(sxy), last, sxy)        # reference ball position, the current or most recently seen one
    ref_next = np.where(np.isnan(sxy), nxt, sxy)
    F["ball_visible"] = vis.astype(np.float32)
    F["age_last_s"] = np.minimum(age_l, 500) / fps
    F["age_next_s"] = np.minimum(age_n, 500) / fps
    F["gap_len_s"] = np.minimum(age_l + age_n, 1000) / fps
    v1 = _disp_speed(sxy, int(round(fps)), fps)
    vs = _disp_speed(sxy, 5, fps)
    F["ball_v1s"] = np.nan_to_num(v1, nan=-1.0)
    F["ball_vshort"] = np.nan_to_num(vs, nan=-1.0)
    F["ball_static"] = (np.nan_to_num(v1, nan=9.0) < 0.5).astype(np.float32)
    # geometry
    rx, ry = ref[:, 0], ref[:, 1]
    inb = (rx >= 0) & (rx <= L) & (ry >= 0) & (ry <= W)
    F["ball_in_pitch"] = np.where(np.isnan(rx), -1.0, inb.astype(np.float32))
    F["ball_out_dist"] = np.nan_to_num(np.maximum.reduce([-rx, rx - L, -ry, ry - W, np.zeros(T)]), nan=-1.0)
    F["ball_side_dist"] = np.nan_to_num(np.minimum(ry, W - ry), nan=-1.0)
    F["ball_end_dist"] = np.nan_to_num(np.minimum(rx, L - rx), nan=-1.0)
    F["ball_in_box"] = np.nan_to_num(((np.minimum(rx, L - rx) < 16.5) & (np.abs(ry - W / 2) < 20.16)).astype(np.float32), nan=-1.0)
    dirs = _attack_dir(M, blocks)
    hd = np.array([dirs.get(int(s), 1) for s in seg])
    F["ball_x_home_dir"] = np.nan_to_num(np.where(hd > 0, rx, L - rx), nan=-1.0)
    # player and ball
    for side, code in (("h", 1), ("a", 2)):
        Pt = P[:, st == code, :]
        for tag, R in (("", ref), ("_next", ref_next)):
            d = np.linalg.norm(Pt - R[:, None, :].astype(np.float32), axis=2)
            d = np.where(np.isnan(d), 99.0, d)
            ds = np.sort(d, axis=1)
            F["d%s1%s" % (side, tag)] = ds[:, 0]
            if tag == "":
                F["d%s2" % side] = ds[:, 1] if ds.shape[1] > 1 else ds[:, 0]
                F["n%s_3m" % side] = (d < 3).sum(1).astype(np.float32)
                F["n%s_8m" % side] = (d < 8).sum(1).astype(np.float32)
        # team shape
        okp = ~np.isnan(Pt[:, :, 0])
        F["n_%s" % side] = okp.sum(1).astype(np.float32)
        with np.errstate(all="ignore"):
            cx, cy = np.nanmean(Pt[:, :, 0], 1), np.nanmean(Pt[:, :, 1], 1)
            F["c%s_x_dir" % side] = np.nan_to_num(np.where(hd > 0, cx, L - cx), nan=-1.0)
            F["c%s_y" % side] = np.nan_to_num(cy, nan=-1.0)
            F["spread_%s" % side] = np.nan_to_num(np.nanstd(Pt[:, :, 0], 1) + np.nanstd(Pt[:, :, 1], 1), nan=-1.0)
            sp = _smooth(Pt.reshape(T, -1), smooth_w).reshape(T, -1, 2)
            d1 = np.full((T, Pt.shape[1]), np.nan)
            lag = int(round(fps))
            d1[lag:] = np.linalg.norm(sp[lag:] - sp[:-lag], axis=2) * fps / lag
            F["speed_%s" % side] = np.nan_to_num(np.nanmean(d1, 1), nan=-1.0)
            F["run_frac_%s" % side] = np.nan_to_num(np.nanmean(d1 > 2.0, 1), nan=-1.0)
            # foremost player (along the home attacking direction)
            xd = np.where(hd[:, None] > 0, Pt[:, :, 0], L - Pt[:, :, 0])
            F["front_%s" % side] = np.nan_to_num(np.nanmax(xd, 1), nan=-1.0)
            F["back_%s" % side] = np.nan_to_num(np.nanmin(xd, 1), nan=-1.0)
    F["d_diff"] = F["dh1"] - F["da1"]
    F["d_diff_next"] = F["dh1_next"] - F["da1_next"]
    F["nearest_home"] = np.where(vis, (F["dh1"] <= F["da1"]).astype(np.float32), -1.0)
    F["near_home_vis"] = ((F["dh1"] <= F["da1"]) & vis).astype(np.float32)
    F["near_away_vis"] = ((F["dh1"] > F["da1"]) & vis).astype(np.float32)
    F["speed_diff"] = F["speed_h"] - F["speed_a"]
    F["speed_all"] = (F["speed_h"] + F["speed_a"]) / 2
    F["players_slow"] = (F["speed_all"] < 1.2).astype(np.float32)
    F["c_diff_x"] = F["ch_x_dir"] - F["ca_x_dir"]
    # short time scales, for short possession spells. Direction change and acceleration of the ball (traces of a touch), time since the nearest team last flipped, ratio of the nearest distances of the two teams
    v5 = np.full((T, 2), np.nan)
    v5[5:] = (sxy[5:] - sxy[:-5]) * (fps / 5)
    ang = np.arctan2(v5[:, 1], v5[:, 0])
    dang = np.full(T, np.nan)
    dang[5:] = np.abs(np.angle(np.exp(1j * (ang[5:] - ang[:-5]))))
    F["ball_turn_5"] = np.nan_to_num(dang, nan=-1.0)
    sp5 = np.linalg.norm(v5, axis=1)
    acc = np.full(T, np.nan)
    acc[5:] = (sp5[5:] - sp5[:-5]) * (fps / 5)
    F["ball_acc_5"] = np.nan_to_num(acc, nan=0.0)
    nt_vis = np.where(vis, np.where(F["dh1"] <= F["da1"], 1, 2), 0)
    flip = np.zeros(T, bool)
    last = 0
    for i in range(T):
        if nt_vis[i] and nt_vis[i] != last:
            flip[i] = last != 0
            last = nt_vis[i]
        if i > 0 and seg[i] != seg[i - 1]:
            last = 0
    age_flip = np.full(T, 10 ** 6)
    j = -1
    for i in range(T):
        if i > 0 and seg[i] != seg[i - 1]:
            j = -1
        if flip[i]:
            j = i
        age_flip[i] = i - j if j >= 0 else 10 ** 6
    F["nearest_flip_age_s"] = np.minimum(age_flip, 250) / fps
    F["d_ratio"] = F["dh1"] / np.maximum(F["dh1"] + F["da1"], 1e-3)
    # time windows
    win_keys = ["near_home_vis", "near_away_vis", "ball_visible", "players_slow", "ball_static", "nh_3m", "na_3m", "speed_diff", "ball_out_dist"]
    for k in win_keys:
        x = F[k].astype(np.float64)
        for w_s in (1, 3, 10):
            w = int(round(w_s * fps))
            F["%s_past%d" % (k, w_s)] = window_mean(x, w, seg)
            F["%s_fut%d" % (k, w_s)] = window_mean(x, w, seg, True)
    for k in ("near_home_vis", "near_away_vis", "ball_turn_5"):
        x = np.maximum(F[k].astype(np.float64), 0)
        for w_s in (0.2, 0.5):
            w = int(round(w_s * fps))
            F["%s_past%gs" % (k, w_s)] = window_mean(x, w, seg)
            F["%s_fut%gs" % (k, w_s)] = window_mean(x, w, seg, True)
    if gap_feats:
        gap_features(M, {"hd": hd}, F, sxy, seg, blocks, fps)
    if touch_feats:
        touch_frame_features(M, F, blocks, fps, bxy, touch_set=touch_set)
    if kin_feats:
        kin_features(F, sxy, ref, P, st, vis, fps)
    if tta_feats:
        tta_features(F, ref, P, st, fps)
    if setp_feats:
        setp_features(F, P, st, hd, L, W, fps, seg, clip=(setp_feats == "clip"))   # with setp_feats set to "clip" outliers are clipped
    if shape_feats:
        shape_features(F, P, st, hd, L, W, fps, seg)
    if press_feats:
        press_features(F, P, st, hd, L, W, fps, seg, ref, ref_next)
    if roster_feats:
        roster_features(F, P, st, hd, L, W, fps)
    if dir_feats:
        dir_features(F, P, st, hd, L, W, fps, seg)
    if comove_feats:                                  # off by default
        comove_features(F, sxy, P, st, vis, fps, seg, v2=(comove_feats == "v2"))   # with "v2" 9 more columns are added
    if pset_feats:                                    # 0 by default. Must come last
        pset_features(F, ref, sxy, P, st, hd, vis, fps, seg, int(pset_feats))
    names = list(F)
    X = np.column_stack([np.asarray(F[k], np.float32) for k in names])
    return X, names
