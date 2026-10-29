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

Gap filling and smoothing
-------------------------
Keypoints with likelihood >= `min_likelihood` are the measurements of a constant-velocity Kalman filter with a
Rauch-Tung-Striebel smoother (one per keypoint coordinate); single-frame jumps (> 40 px from the median of the 7
surrounding frames) are dropped first, and while a point is tracked, detections more than 6 std (>= 30 px) from
the predicted position are rejected (DLC sometimes alternates the LED with another bright spot for several frames).
Below the threshold the keypoint is missing and the
smoother bridges the gap; its posterior std grows with the gap length, and filtered positions with a std above
`max_std` are NaN. The rat and the robot get their own acceleration noise (`rat_filter`, `robot_filter`): on held-out
stretches of a session (cam2), rat gaps of 3/10/30 frames were filled with a median error of ~1/4/9 px, robot gaps of
up to 90 frames with ~2 px. With the defaults rat gaps up to ~0.4 s and robot gaps up to ~3 s are filled. This is the
same idea as DLC's `filterpredictions(filtertype='arima')` (a state-space model), but runs on the analytic directly.

Output columns
--------------
Keypoints:
    <bp>_x, <bp>_y, <bp>_likelihood      raw DLC output, unchanged
    <bp>_x_filt, <bp>_y_filt             smoothed and gap-filled, NaN where the posterior std > max_std
    <bp>_filt_std                       posterior std of the filtered position, px
Head (rat):
    head_x, head_y                      logger LED (on the head-mounted logger; the most reliably tracked point), px
    head_speed                          floor-scaled px/s
    head_direction_dlc                  perpendicular to the left->right ear line, deg
    head_direction_dlc_vel              deg/s
    snout_direction_dlc                 ear midpoint -> nose, deg
    ear_span                            ear distance, image px (shrinks when the head faces toward/away from the camera)
Body (rat):
    body_x, body_y                      midpoint of logger_back (rear of the head-mounted logger) and tail_base, px
    body_direction                      tail_base -> logger_back, deg
    head_body_angle                     head_direction_dlc - body_direction, deg, + = head turned left
    speed                               body midpoint speed, floor-scaled px/s
    movement_energy                     median speed over the tracked rat keypoints, floor-scaled px/s
Robot:
    robot_x, robot_y                    robot centre (midpoint of robot_front and robot_back), px. With only one point
                                        tracked, the other is inferred with the front - back image vector interpolated in
                                        time from the nearest frames with both (the robot turns slowly: median 2 deg/s)
    robot_n_points                      2 = both points tracked, 1 = centre inferred from one point, 0 = no robot
    robot_inferred_age_s                for robot_n_points == 1: time to the nearest frame with both points, s
    robot_direction                     robot_back -> robot_front (measured or interpolated), deg
    robot_speed                         floor-scaled px/s
    robot_turn_rate                     deg/s
Head orientation (IMU, from the HeadOrientation analytic, at the frame times; only if given):
    head_direction, snout_elevation, head_tilt, imu_clean   see logger_head_orientation.py
Rat and robot:
    robot_distance                      head (LED) -> robot centre, floor-scaled px
    robot_bearing                       direction of the robot centre seen from the head relative to the IMU head_direction,
                                        deg, 0 = robot straight ahead, +-180 = behind, + = robot on the left
    robot_bearing_dlc                   the same relative to head_direction_dlc (ear line)
    turn_towards_robot                  True at the onset of a turn towards the robot: |robot_bearing| > 60 deg in >= 80 %
                                        of the preceding 1 s, then < 20 deg in >= 80 % of the following 0.5 s
"""

import warnings

import numpy as np
import pandas as pd

RAT_KEYPOINTS = ['nose', 'left_ear', 'right_ear', 'logger_led', 'logger_back', 'tail_base']
ROBOT_KEYPOINTS = ['robot_front', 'robot_back']


def _wrap(deg):
    return (deg + 180) % 360 - 180

def _direction(dx, dy, floor_y_scale):
    """Floor direction of an image vector (y down), deg, counterclockwise from image right."""
    return np.degrees(np.arctan2(-dy / floor_y_scale, dx))

def _floor_dist(dx, dy, floor_y_scale):
    return np.hypot(dx, dy / floor_y_scale)

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


def kalman_rts(z, ts, accel_std, meas_std, gate=None, min_gate=30.):
    """Constant-velocity Kalman filter and Rauch-Tung-Striebel smoother, independently per column of z.

    Parameters
    ----------
    z : (T, N) array, NaN = missing measurement
    ts : (T,) timestamps in s
    accel_std : std of the (white) acceleration noise, units/s^2; larger = follows faster changes, bridges gaps worse
    meas_std : measurement noise std, units
    gate : if given, measurements further than max(gate * innovation std, min_gate) from the prediction are rejected
        (validation gate: a detection that jumps to another object is ignored while the point is being tracked)

    Returns smoothed positions (T, N) and their posterior std (T, N).
    """
    z = np.asarray(z, float)
    T, N = z.shape
    dt = np.diff(ts, prepend=ts[0] - np.median(np.diff(ts)))
    first = np.argmax(~np.isnan(z), axis=0)
    m = np.zeros((N, 2))
    m[:, 0] = np.where(np.isnan(z).all(0), 0, z[first, np.arange(N)])
    P = np.tile(np.diag([1e8, 1e8]), (N, 1, 1))
    m_pred, P_pred = np.empty((T, N, 2)), np.empty((T, N, 2, 2))
    m_filt, P_filt = np.empty((T, N, 2)), np.empty((T, N, 2, 2))
    R = meas_std ** 2
    for t in range(T):                                       # forward Kalman filter
        d = dt[t]
        F = np.array([[1, d], [0, 1]])
        Q = accel_std ** 2 * np.array([[d ** 4 / 4, d ** 3 / 2], [d ** 3 / 2, d ** 2]])
        m = m @ F.T
        P = F @ P @ F.T + Q
        m_pred[t], P_pred[t] = m, P
        obs = ~np.isnan(z[t])
        if gate is not None:
            with np.errstate(invalid="ignore"):
                obs &= np.abs(z[t] - m[:, 0]) <= np.maximum(gate * np.sqrt(P[:, 0, 0] + R), min_gate)
        if obs.any():
            K = P[obs, :, 0] / (P[obs, 0, 0] + R)[:, None]
            m[obs] = m[obs] + K * (z[t, obs] - m[obs, 0])[:, None]
            P[obs] = P[obs] - K[:, :, None] * P[obs, 0, :][:, None, :]
        m_filt[t], P_filt[t] = m, P
    m_s, P_s = m_filt.copy(), P_filt.copy()
    for t in range(T - 2, -1, -1):                           # backward RTS smoother
        F = np.array([[1, dt[t + 1]], [0, 1]])
        G = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        m_s[t] = m_filt[t] + np.einsum('nij,nj->ni', G, m_s[t + 1] - m_pred[t + 1])
        P_s[t] = P_filt[t] + G @ (P_s[t + 1] - P_pred[t + 1]) @ np.transpose(G, (0, 2, 1))
    return m_s[:, :, 0], np.sqrt(np.maximum(P_s[:, :, 0, 0], 0))


def _jumps(x, y, window, max_dev):
    """Samples further than max_dev from the centred rolling median of their neighbours (single-frame detection errors)."""
    mx = pd.Series(x).rolling(window, center=True, min_periods=3).median().values
    my = pd.Series(y).rolling(window, center=True, min_periods=3).median().values
    return np.hypot(x - mx, y - my) > max_dev


def filter_keypoints(df, ts, bodyparts, min_likelihood, accel_std, meas_std, max_std, jump_window=7, max_jump=40.):
    """Adds <bp>_x_filt, <bp>_y_filt, <bp>_filt_std (see module docstring) for the given bodyparts."""
    z = []
    for bp in bodyparts:
        ok = df[f"{bp}_likelihood"].values >= min_likelihood
        x, y = np.where(ok, df[f"{bp}_x"].values, np.nan), np.where(ok, df[f"{bp}_y"].values, np.nan)
        bad = _jumps(x, y, jump_window, max_jump)
        z += [np.where(bad, np.nan, x), np.where(bad, np.nan, y)]
    pos, std = kalman_rts(np.stack(z, 1), ts, accel_std, meas_std, gate=6.)
    for i, bp in enumerate(bodyparts):
        s = np.maximum(std[:, 2 * i], std[:, 2 * i + 1])
        keep = s <= max_std
        df[f"{bp}_x_filt"] = np.where(keep, pos[:, 2 * i], np.nan)
        df[f"{bp}_y_filt"] = np.where(keep, pos[:, 2 * i + 1], np.nan)
        df[f"{bp}_filt_std"] = s


def add_arena_features(
    df: pd.DataFrame,
    timestamps_us: np.ndarray,
    min_likelihood: float = 0.6,
    floor_y_scale: float = 0.69,
    rat_filter: tuple = (2000., 3., 10.),
    robot_filter: tuple = (200., 3., 15.),
    smooth_frames: int = 5,
    max_angular_vel: float = 1500.0,
    head_orientation: pd.DataFrame = None,
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
        Keypoints below this likelihood are treated as missing.
    floor_y_scale : float
        Foreshortening of the floor's depth axis in the image (image dy / true dy).
    rat_filter, robot_filter : (accel_std px/s^2, meas_std px, max_std px)
        Kalman smoother settings for the rat and the robot keypoints, see kalman_rts.
    smooth_frames : int
        Centered rolling mean (frames) applied to angles before angular velocities.
    max_angular_vel : float
        Angular velocities above this (deg/s) are treated as tracking errors and set to NaN.
    head_orientation : pd.DataFrame, optional
        HeadOrientation analytic (ephys_timestamp, head_direction, snout_elevation, head_tilt, clean); adds the IMU columns,
        robot_bearing and turn_towards_robot.
    """
    df = df.copy()
    ts = np.asarray(timestamps_us, dtype=float) / 1e6
    rat = [bp for bp in RAT_KEYPOINTS if f"{bp}_likelihood" in df]
    robot = [bp for bp in ROBOT_KEYPOINTS if f"{bp}_likelihood" in df]

    # ── 1. Smooth and fill gaps, per keypoint ─────────────────────────────
    filter_keypoints(df, ts, rat, min_likelihood, *rat_filter)
    filter_keypoints(df, ts, robot, min_likelihood, *robot_filter)
    x = lambda bp: df[f"{bp}_x_filt"].values
    y = lambda bp: df[f"{bp}_y_filt"].values
    speed = lambda px, py: _floor_dist(_deriv(px, ts), _deriv(py, ts), floor_y_scale)

    # ── 2. Head ───────────────────────────────────────────────────────────
    df["head_x"], df["head_y"] = x("logger_led"), y("logger_led")
    df["head_speed"] = speed(df.head_x.values, df.head_y.values)
    ear_dx, ear_dy = x("right_ear") - x("left_ear"), y("right_ear") - y("left_ear")
    # forward = left->right ear vector rotated by +90 deg (counterclockwise, floor frame)
    df["head_direction_dlc"] = np.degrees(np.arctan2(ear_dx, ear_dy / floor_y_scale))
    df["head_direction_dlc_vel"] = _angular_vel(df["head_direction_dlc"].values, ts, smooth_frames, max_angular_vel)
    ears_x, ears_y = (x("left_ear") + x("right_ear")) / 2, (y("left_ear") + y("right_ear")) / 2
    df["snout_direction_dlc"] = _direction(x("nose") - ears_x, y("nose") - ears_y, floor_y_scale)
    df["ear_span"] = np.hypot(ear_dx, ear_dy)

    # ── 3. Body ───────────────────────────────────────────────────────────
    df["body_x"] = (x("logger_back") + x("tail_base")) / 2
    df["body_y"] = (y("logger_back") + y("tail_base")) / 2
    df["body_direction"] = _direction(x("logger_back") - x("tail_base"), y("logger_back") - y("tail_base"),
                                      floor_y_scale)
    df["head_body_angle"] = _wrap(df["head_direction_dlc"].values - df["body_direction"].values)
    df["speed"] = speed(df.body_x.values, df.body_y.values)
    kp_speeds = np.stack([speed(x(bp), y(bp)) for bp in rat], 1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                     # all-NaN frames
        df["movement_energy"] = np.nanmedian(kp_speeds, axis=1)

    # ── 4. Robot: centre from both points, or from one point + the interpolated front - back vector ──
    fx, fy, bx, by = x("robot_front"), y("robot_front"), x("robot_back"), y("robot_back")
    both = ~np.isnan(fx) & ~np.isnan(bx)
    vx, vy = (pd.Series(np.where(both, v, np.nan), index=ts).interpolate(method='index', limit_direction='both').values
              for v in (fx - bx, fy - by))
    df["robot_x"] = np.where(both, (fx + bx) / 2, np.where(~np.isnan(bx), bx + vx / 2, fx - vx / 2))
    df["robot_y"] = np.where(both, (fy + by) / 2, np.where(~np.isnan(by), by + vy / 2, fy - vy / 2))
    n_points = np.where(both, 2, np.where(np.isnan(df.robot_x.values), 0, 1)).astype(np.int8)
    df["robot_n_points"] = n_points
    t_both = pd.Series(np.where(both, ts, np.nan))
    age = np.fmin(np.abs(ts - t_both.ffill().values), np.abs(t_both.bfill().values - ts))
    df["robot_inferred_age_s"] = np.where(n_points == 1, age, np.nan)
    df["robot_direction"] = np.where(n_points > 0, _direction(vx, vy, floor_y_scale), np.nan)
    df["robot_speed"] = speed(df.robot_x.values, df.robot_y.values)
    df["robot_turn_rate"] = _angular_vel(df["robot_direction"].values, ts, smooth_frames, max_angular_vel)

    # ── 5. Head orientation (IMU) at the frame times ──────────────────────
    to_robot_dx, to_robot_dy = df.robot_x.values - df.head_x.values, df.robot_y.values - df.head_y.values
    to_robot = _direction(to_robot_dx, to_robot_dy, floor_y_scale)
    df["robot_distance"] = _floor_dist(to_robot_dx, to_robot_dy, floor_y_scale)
    df["robot_bearing_dlc"] = _wrap(to_robot - df["head_direction_dlc"].values)
    if head_orientation is not None:
        t_imu = head_orientation.ephys_timestamp.values / 1e6
        df["head_direction"] = _wrap(np.degrees(np.interp(ts, t_imu, np.unwrap(np.deg2rad(head_orientation.head_direction.values)))))
        for c in ("snout_elevation", "head_tilt"):
            df[c] = np.interp(ts, t_imu, head_orientation[c].values)
        df["imu_clean"] = head_orientation.clean.values[np.clip(np.searchsorted(t_imu, ts), 0, len(t_imu) - 1)]
        df["robot_bearing"] = _wrap(to_robot - df["head_direction"].values)
        df["turn_towards_robot"] = _turn_onsets(np.abs(df["robot_bearing"].values))
    return df


def _turn_onsets(abs_bearing, far=60., near=20., before=30, after=15, frac=.8, min_gap=90):
    """Onsets of turns towards the robot (see turn_towards_robot in the module docstring); NaN bearings are ignored,
    at least half of each window must be defined."""
    def window_frac(cond, n, ahead):
        v = pd.Series(np.where(np.isnan(abs_bearing), np.nan, cond.astype(float)))
        v = v[::-1] if ahead else v
        r = v.rolling(n, min_periods=n // 2).mean()
        return np.nan_to_num((r[::-1] if ahead else r.shift(1)).values)
    with np.errstate(invalid="ignore"):
        on = np.flatnonzero((window_frac(abs_bearing > far, before, False) >= frac) &
                            (window_frac(abs_bearing < near, after, True) >= frac))
    on = on[np.r_[True, np.diff(on) > min_gap]] if len(on) else on
    onset = np.zeros(len(abs_bearing), bool)
    onset[on] = True
    return onset


def arena_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """NaN%, median and range of the computed feature columns."""
    cols = [c for c in df.columns if c.split('_')[0] in ('head', 'snout', 'ear', 'body', 'speed', 'movement', 'robot')
            and not c.startswith(('robot_front', 'robot_back'))]
    rows = [{"feature": c, "nan_pct": round(100 * df[c].isna().mean(), 1), "median": round(df[c].median(), 2),
             "min": round(df[c].min(), 2), "max": round(df[c].max(), 2)} for c in cols]
    return pd.DataFrame(rows).set_index("feature")
