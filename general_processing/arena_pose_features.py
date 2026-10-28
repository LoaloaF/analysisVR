"""
arena_pose_features.py
======================
Kinematic features from DeepLabCut keypoints of the free-moving logger arena sessions (model and camera from
general_processing/dlc_arena.py, sessions analyzed on cam2). Unlike pose_features.py (head-fixed animals), the rat
moves through the arena, so positions, speed and the robot's position relative to the rat are the main features.

Coordinates and angles
----------------------
Positions are image px of the analyzed camera (x right, y down). The camera looks obliquely onto the floor, which
squashes the floor's depth axis; image y differences are divided by `floor_y_scale` (cam2: 0.69, fitted from the LED
walking direction against the magnetometer heading) before directions, distances and speeds are computed.
Directions follow the IMU `head_direction` convention: degrees in [-180, 180), counterclockwise from image right as
seen from above, 90 = away from the camera, + = turning left. So `head_direction_dlc` and the IMU `head_direction`
can be compared directly (the DLC one does not drift, the IMU one is smoother and works when the head is hidden).

`head_direction_dlc` is perpendicular to the left->right ear line. On hand labels it matched the IMU head direction
to ~5 deg (median, cam2), better than the ear-midpoint->nose direction (`snout_direction_dlc`, ~10 deg), which
shifts when the head pitches. Left/right ear swaps flip it by 180 deg.

Keypoints below `min_likelihood` are NaN, gaps up to `max_gap` frames are interpolated linearly.

Output columns
--------------
Keypoints:
    <bp>_x, <bp>_y, <bp>_likelihood      x/y NaN below min_likelihood (likelihood kept)
Head:
    head_x, head_y                      ear midpoint, px
    head_direction_dlc                  perpendicular to the left->right ear line, deg
    head_direction_dlc_vel              deg/s
    snout_direction_dlc                 ear midpoint -> nose, deg
    ear_span                            ear distance, image px (shrinks when the head faces toward/away from the camera)
Body:
    body_x, body_y                      midpoint of logger_back (rear of the head-mounted logger) and tail_base, px
    body_direction                      tail_base -> logger_back, deg
    head_body_angle                     head_direction_dlc - body_direction, deg, + = head turned left
    speed                               body midpoint speed, floor-scaled px/s
Robot:
    robot_x, robot_y                    midpoint of robot_front and robot_back, px
    robot_direction                     robot_back -> robot_front, deg
    robot_distance                      head -> robot midpoint, floor-scaled px
    robot_bearing                       direction of the robot seen from the head, relative to head_direction_dlc,
                                        deg, 0 = straight ahead, + = to the left
"""

import numpy as np
import pandas as pd


def _wrap(deg):
    return (deg + 180) % 360 - 180

def _direction(dx, dy, floor_y_scale):
    """Floor direction of an image vector (y down), deg, counterclockwise from image right."""
    return np.degrees(np.arctan2(-dy / floor_y_scale, dx))

def _floor_dist(dx, dy, floor_y_scale):
    return np.hypot(dx, dy / floor_y_scale)

def _fill_short_gaps(values, max_gap):
    """Linear interpolation of NaN runs of at most max_gap samples; longer runs stay NaN entirely."""
    s = pd.Series(values)
    isna = s.isna()
    run_len = isna.groupby((~isna).cumsum()).transform('sum')
    return s.interpolate(limit_area='inside').where(~isna | (run_len <= max_gap)).values

def _smooth(values, n):
    return pd.Series(values).rolling(n, center=True, min_periods=1).mean().values

def _deriv(values, ts):
    return np.gradient(values, ts)

def _angular_vel(angle, ts, smooth_frames, max_dps):
    valid = ~np.isnan(angle)
    unwrapped = np.full(len(angle), np.nan)
    unwrapped[valid] = np.unwrap(np.deg2rad(angle[valid]))     # np.unwrap propagates NaNs, so only the valid samples
    vel = np.rad2deg(_deriv(_smooth(unwrapped, smooth_frames), ts))
    return np.where(np.abs(vel) > max_dps, np.nan, vel)


def add_arena_features(
    df: pd.DataFrame,
    timestamps_us: np.ndarray,
    min_likelihood: float = 0.6,
    max_gap: int = 5,
    floor_y_scale: float = 0.69,
    smooth_frames: int = 5,
    max_angular_vel: float = 1500.0,
) -> pd.DataFrame:
    """
    Compute and append the arena features (see module docstring).

    Parameters
    ----------
    df : pd.DataFrame
        DLC output with <bp>_x, <bp>_y, <bp>_likelihood columns for the dlc_arena BODYPARTS, one row per frame.
    timestamps_us : np.ndarray
        Frame timestamps in us (ephys time).
    min_likelihood : float
        Keypoints below this likelihood become NaN.
    max_gap : int
        Longest run of NaN frames (per keypoint) that is interpolated.
    floor_y_scale : float
        Foreshortening of the floor's depth axis in the image (image dy / true dy).
    smooth_frames : int
        Centered rolling mean (frames) applied before velocities.
    max_angular_vel : float
        Angular velocities above this (deg/s) are treated as tracking errors and set to NaN.
    """
    df = df.copy()
    ts = np.asarray(timestamps_us, dtype=float) / 1e6
    bodyparts = [c[:-len('_likelihood')] for c in df.columns if c.endswith('_likelihood')]

    # ── 1. Mask low-likelihood keypoints, fill short gaps ─────────────────
    for bp in bodyparts:
        below = df[f"{bp}_likelihood"].values < min_likelihood
        for c in (f"{bp}_x", f"{bp}_y"):
            df[c] = _fill_short_gaps(np.where(below, np.nan, df[c].values), max_gap)
    x = lambda bp: df[f"{bp}_x"].values
    y = lambda bp: df[f"{bp}_y"].values

    # ── 2. Head ───────────────────────────────────────────────────────────
    df["head_x"] = (x("left_ear") + x("right_ear")) / 2
    df["head_y"] = (y("left_ear") + y("right_ear")) / 2
    ear_dx, ear_dy = x("right_ear") - x("left_ear"), y("right_ear") - y("left_ear")
    # forward = left->right ear vector rotated by +90 deg (counterclockwise, floor frame)
    df["head_direction_dlc"] = np.degrees(np.arctan2(ear_dx, ear_dy / floor_y_scale))
    df["head_direction_dlc_vel"] = _angular_vel(df["head_direction_dlc"].values, ts, smooth_frames, max_angular_vel)
    df["snout_direction_dlc"] = _direction(x("nose") - df.head_x.values, y("nose") - df.head_y.values, floor_y_scale)
    df["ear_span"] = np.hypot(ear_dx, ear_dy)

    # ── 3. Body ───────────────────────────────────────────────────────────
    df["body_x"] = (x("logger_back") + x("tail_base")) / 2
    df["body_y"] = (y("logger_back") + y("tail_base")) / 2
    df["body_direction"] = _direction(x("logger_back") - x("tail_base"), y("logger_back") - y("tail_base"),
                                      floor_y_scale)
    df["head_body_angle"] = _wrap(df["head_direction_dlc"].values - df["body_direction"].values)
    vx = _deriv(_smooth(df.body_x.values, smooth_frames), ts)
    vy = _deriv(_smooth(df.body_y.values, smooth_frames), ts)
    df["speed"] = _floor_dist(vx, vy, floor_y_scale)

    # ── 4. Robot ──────────────────────────────────────────────────────────
    df["robot_x"] = (x("robot_front") + x("robot_back")) / 2
    df["robot_y"] = (y("robot_front") + y("robot_back")) / 2
    df["robot_direction"] = _direction(x("robot_front") - x("robot_back"), y("robot_front") - y("robot_back"),
                                       floor_y_scale)
    to_robot_dx, to_robot_dy = df.robot_x.values - df.head_x.values, df.robot_y.values - df.head_y.values
    df["robot_distance"] = _floor_dist(to_robot_dx, to_robot_dy, floor_y_scale)
    df["robot_bearing"] = _wrap(_direction(to_robot_dx, to_robot_dy, floor_y_scale) - df["head_direction_dlc"].values)
    return df


def arena_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """NaN%, median and range of the computed feature columns."""
    cols = [c for c in df.columns if c.split('_')[0] in ('head', 'snout', 'ear', 'body', 'speed', 'robot')
            and not c.startswith(('robot_front', 'robot_back'))]
    rows = [{"feature": c, "nan_pct": round(100 * df[c].isna().mean(), 1), "median": round(df[c].median(), 2),
             "min": round(df[c].min(), 2), "max": round(df[c].max(), 2)} for c in cols]
    return pd.DataFrame(rows).set_index("feature")
