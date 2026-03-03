"""
pose_kinematics.py
==================
Systematic kinematic feature extraction from DeepLabCut pose estimation data.
Designed for head-fixed / harness-restrained animals where translational
body movement is absent — all features are postural or rotational.

Column naming
-------------
Pass ``segment_names`` — a dict mapping (p1, p2) keypoint tuples to short
human-readable names — to get clean column names:

    segment_names = {
        ('nose', 'neck'):                 'head',
        ('neck', 'right_front_paw'):      'front_paw',
        ('neck', 'mid_spine'):            'fixed_body',
        ('mid_spine', 'late_spine'):      'lower_body',
        ('late_spine', 'right_hind_paw'): 'hind_paw',
    }

Output columns (with segment_names)
------------------------------------
Per segment:
    <n>_length          px distance
    <n>_angle           absolute heading, deg [0, 360)
    <n>_angle_vel       heading angular velocity, deg/s
    <n>_likelihood      min(lk_p1, lk_p2); NaN when either is below threshold

Per joint:
    <seg_a>_<seg_b>_bend          opening angle at joint, deg [0, 180]
    <seg_a>_<seg_b>_bend_vel      angular velocity, deg/s
    <seg_a>_<seg_b>_bend_likelihood

Body composites:
    spine_curvature               mean bend across all joints
    body_elongation               end-to-end / sum-of-segment lengths
    head_body_angle               head angle minus body angle, signed [-180, 180]

Movement energy:
    movement_energy               RMS of all bend velocities per frame
    movement_energy_smooth5       5-frame rolling mean
"""

import numpy as np
import pandas as pd
from typing import Optional


def _vec(df, p1, p2):
    return df[f"{p2}_x"].values - df[f"{p1}_x"].values, \
           df[f"{p2}_y"].values - df[f"{p1}_y"].values

def _length(df, p1, p2):
    return np.hypot(*_vec(df, p1, p2))

def _orientation(df, p1, p2):
    dx, dy = _vec(df, p1, p2)
    return np.degrees(np.arctan2(dy, dx)) % 360

def _joint_angle(df, arm1, vertex, arm2):
    dx1, dy1 = _vec(df, vertex, arm1)
    dx2, dy2 = _vec(df, vertex, arm2)
    n1, n2   = np.hypot(dx1, dy1), np.hypot(dx2, dy2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = (dx1*dx2 + dy1*dy2) / (n1*n2)
    return np.where((n1==0)|(n2==0), np.nan,
                    np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def _deriv(values, ts):
    dt = np.diff(ts)
    r  = np.where(dt > 0, np.diff(values) / dt, np.nan)
    return np.concatenate(([r[0]], r))

def _ang_vel(angle, ts, max_dps):
    return np.where(np.abs(v := _deriv(angle, ts)) > max_dps, np.nan, v)

def _valid_mask(df, points, threshold):
    mask = np.ones(len(df), dtype=bool)
    for p in points:
        col = f"{p}_likelihood"
        if col in df.columns:
            mask &= df[col].values >= threshold
    return mask

def _min_lk(df, points):
    cols = [f"{p}_likelihood" for p in points if f"{p}_likelihood" in df.columns]
    return np.vstack([df[c].values.astype(float) for c in cols]).min(axis=0) if cols \
           else np.ones(len(df))


def add_kinematic_features(
    df: pd.DataFrame,
    skeleton: list[list[str]],
    timestamps_us: np.ndarray,
    min_likelihood: float = 0.6,
    max_angular_vel: float = 720.0,
    segment_names: Optional[dict] = None,
    extra_joints: Optional[list[list[str]]] = None,
) -> pd.DataFrame:
    """
    Compute and append postural/rotational kinematic features.

    Parameters
    ----------
    df : pd.DataFrame
        Pose data with <point>_x, <point>_y, <point>_likelihood columns.
    skeleton : list of [p1, p2]
        Ordered segment chain from head to tail.
    timestamps_us : np.ndarray
        Frame timestamps (auto-detects µs / ms / s from median inter-frame gap).
    min_likelihood : float
        Frames where any required keypoint is below this become NaN.
    max_angular_vel : float
        Angular velocity outlier clip, deg/s.
    segment_names : dict {(p1,p2): name}, optional
        Human-readable names for segments, used in all output column names.
        Falls back to '<p1>_<p2>' if a segment is not found in the dict.
    extra_joints : list of [arm1, vertex, arm2], optional
        Additional joints beyond those auto-detected from consecutive segments.
        Needed for branching skeletons (e.g. neck has two outgoing segments).
    """
    df       = df.copy()
    seg_names = {tuple(k): v for k, v in (segment_names or {}).items()}
    segments  = [tuple(s) for s in skeleton]

    def sn(p1, p2):
        """Short name for segment (p1, p2)."""
        return seg_names.get((p1, p2), f"{p1}_{p2}")

    # ── timestamps → seconds ─────────────────────────────────────────────
    med = np.nanmedian(np.diff(timestamps_us))
    ts  = timestamps_us / (1e6 if med > 1_000 else 1e3 if med > 0.5 else 1)

    # ── 1. Segment-level features ─────────────────────────────────────────
    for p1, p2 in segments:
        name  = sn(p1, p2)
        valid = _valid_mask(df, [p1, p2], min_likelihood)
        lk_v  = _min_lk(df, [p1, p2])
        ori   = _orientation(df, p1, p2)
        ori_v = _ang_vel(ori, ts, max_angular_vel)   # computed pre-mask

        df[f"{name}_length"]     = np.where(valid, _length(df, p1, p2), np.nan)
        df[f"{name}_angle"]      = np.where(valid, ori,   np.nan)
        df[f"{name}_angle_vel"]  = np.where(valid, ori_v, np.nan)
        df[f"{name}_likelihood"] = np.where(valid, lk_v,  np.nan)

    # ── 2. Joint bend features ────────────────────────────────────────────
    bend_cols, bend_vel_cols = [], []

    def _add_bend(arm1, vertex, arm2):
        # Name the joint by the two segment names that meet at vertex
        # Incoming segment: arm1 → vertex  |  Outgoing: vertex → arm2
        tag   = f"{sn(arm1, vertex)}_{sn(vertex, arm2)}_bend"
        if tag in df.columns:
            return  # already computed (extra_joints dedup)
        valid = _valid_mask(df, [arm1, vertex, arm2], min_likelihood)
        lk_v  = _min_lk(df, [arm1, vertex, arm2])
        angle = _joint_angle(df, arm1, vertex, arm2)
        vel   = _ang_vel(angle, ts, max_angular_vel)
        df[tag]                 = np.where(valid, angle, np.nan)
        df[f"{tag}_vel"]        = np.where(valid, vel,   np.nan)
        df[f"{tag}_likelihood"] = np.where(valid, lk_v,  np.nan)
        bend_cols.append(tag)
        bend_vel_cols.append(f"{tag}_vel")

    # Consecutive segments share a vertex → auto-detect
    for i in range(len(segments) - 1):
        p1a, vtx = segments[i]
        vtxb, p2b = segments[i+1]
        if vtx == vtxb:
            _add_bend(p1a, vtx, p2b)

    # Extra joints (branching skeletons)
    for arm1, vertex, arm2 in (extra_joints or []):
        _add_bend(arm1, vertex, arm2)

    # ── 3. Body composites ────────────────────────────────────────────────
    if bend_cols:
        df["spine_curvature"] = df[bend_cols].mean(axis=1, skipna=True)

    # Body elongation — use most distal keypoint tracked ≥ 20 % of time
    all_kps    = [segments[0][0]] + [s[1] for s in segments]
    body_start = all_kps[0]
    body_end   = next(
        (pt for pt in reversed(all_kps)
         if f"{pt}_likelihood" in df.columns
         and (df[f"{pt}_likelihood"].values >= min_likelihood).mean() >= 0.20),
        all_kps[-1]
    )
    seg_sum = pd.concat(
        [df[f"{sn(p1,p2)}_length"].fillna(0) for p1, p2 in segments], axis=1
    ).sum(axis=1).values
    e2e         = _length(df, body_start, body_end)
    valid_elong = _valid_mask(df, [body_start, body_end], min_likelihood)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["body_elongation"] = np.where(
            valid_elong & (seg_sum > 0), e2e / seg_sum, np.nan
        )

    # Head–body angle (signed [-180, 180])
    if len(segments) >= 2:
        h_ori    = df[f"{sn(*segments[0])}_angle"].values
        b_ori    = df[f"{sn(*segments[1])}_angle"].values
        valid_hb = _valid_mask(df, list(segments[0]) + list(segments[1]),
                               min_likelihood)
        df["head_body_angle"] = np.where(
            valid_hb, (h_ori - b_ori + 180) % 360 - 180, np.nan
        )

    # ── 4. Movement energy ────────────────────────────────────────────────
    if bend_vel_cols:
        vm = df[bend_vel_cols].values.astype(float)
        with np.errstate(invalid="ignore"):
            energy = np.sqrt(np.nanmean(vm**2, axis=1))
        energy[np.all(np.isnan(vm), axis=1)] = np.nan
        df["movement_energy"]        = energy
        df["movement_energy_smooth5"] = (
            pd.Series(energy).rolling(5, center=True, min_periods=1).mean().values
        )

    # ── 5. Mask raw keypoint coordinates below threshold ─────────────────
    # Set <kp>_x, <kp>_y, and <kp>_likelihood to NaN for frames where the
    # keypoint's own likelihood is below min_likelihood.  This ensures any
    # downstream code (animation, plotting) that reads raw coordinates sees
    # NaN rather than uncertain detections.
    all_kp = list({p for seg in segments for p in seg})
    for kp in all_kp:
        lk_col = f"{kp}_likelihood"
        if lk_col not in df.columns:
            continue
        below = df[lk_col].values < min_likelihood
        for col in [f"{kp}_x", f"{kp}_y", lk_col]:
            if col in df.columns:
                df.loc[below, col] = np.nan

    return df


def kinematic_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """NaN%, mean, std, min, max for every computed kinematic column."""
    raw_kp_lk = {f"{kp}_likelihood"
                 for kp in ["nose","neck","right_front_paw",
                             "mid_spine","late_spine","right_hind_paw"]}
    skip_sfx  = ("_x", "_y")
    skip_exact = {"ephys_timestamp","pc_timestamp",
                  "facecam_image_pc_timestamp","facecam_image_ephys_timestamp"}
    cols = [c for c in df.columns
            if not c.endswith(skip_sfx) and c not in skip_exact
            and c not in raw_kp_lk]
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({"feature": col,
                     "nan_pct": round(100*s.isna().mean(), 1),
                     "mean":    round(s.mean(), 4),
                     "std":     round(s.std(),  4),
                     "min":     round(s.min(),  4),
                     "max":     round(s.max(),  4)})
    return pd.DataFrame(rows).set_index("feature")