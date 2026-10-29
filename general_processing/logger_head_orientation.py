"""
logger_head_orientation.py
==========================
Head orientation of the wireless-logger sessions from the logger's IMU, as anatomical, drift-corrected angles.

Sensors and axes
----------------
The `orientation` modality (100 Hz) holds two fusions of the logger IMU. Use the 6-axis one (`orientation_*_imu`,
`orientation_q_imu_*`): it is the integrated gyroscope with pitch/roll anchored to gravity. The 9-axis one is fused with a
magnetometer whose y axis is flipped relative to the IMU and should not be used.

The logger sits on the head turned by 90 deg: logger x = left ear, logger y = towards the tail (snout = -y), z = crown
(right-handed). So `roll_imu` is (minus) the snout elevation and `pitch_imu` is the sideways head tilt; `yaw_imu` is the
direction of the left ear. Shown on the data (2026-09-09 rPG007 session): the head nods about logger x (largest gyro
variance), and video frames at the `roll_imu` extremes show the snout down (+90) or the rat rearing nose-up (-75).

Yaw drift
---------
The 6-axis yaw is relative and drifts (~25 deg/min). The magnetometer (calibrated with the extractor's A, b from
`_original/orientation_rpy.mat`, y axis flipped) is rotated into the 6-axis world frame; within 10 s windows the horizontal
field direction gives the yaw drift, which is interpolated and subtracted (windows with a disturbed field, resultant < 0.8,
are skipped). The corrected yaw is then expressed in the cam2 image frame with `cam2_offset` (fitted from the rat's walking
direction in cam2; it contains the 90 deg between the left ear and the snout). Against the video (DLC ear line) the result
agrees to a median of ~10 deg without trend over the session.

Output columns (one row per orientation sample, 100 Hz)
--------------------------------------------------------
    sample_id, ephys_timestamp     orientation sample id and ephys time (us)
    head_direction                 horizontal direction of the snout, deg [-180, 180), cam2 frame: 0 = image right,
                                   90 = away from cam2, + = turning left. Drift-corrected.
    snout_elevation                elevation of the snout axis above the horizon, deg [-90, 90], + = nose up
                                   (= -roll_imu while the head is level sideways)
    head_tilt                      elevation of the left ear (ear-to-ear line vs the horizon), deg [-90, 90],
                                   + = right ear down (= -pitch_imu exactly)
    <angle>_vel                    time derivatives, deg/s (50 ms smoothing)
    yaw_drift                      the drift that was removed from yaw_imu, deg (diagnostic)
    ephys_patched                  sample interpolated by the extractor
    near_zero_pad                  within 100 ms of a zero pad (spurious or real data loss)
    in_malfunction_window          logger malfunction window
    clean                          not near_zero_pad and not in_malfunction_window
"""
import os

import h5py
import numpy as np
import pandas as pd

from CustomLogger import CustomLogger as Logger


def _wrap(deg):
    return (deg + 180) % 360 - 180

def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

def _rotate(q, v):
    """Rotate body-frame vectors v by quaternions q (w, x, y, z): world = R(q) @ v."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return _qmul(_qmul(q, qv), q * np.array([1., -1, -1, -1]))[..., 1:]

def _in_intervals(sample_ids, starts, ends):
    order = np.argsort(starts)
    a, b = starts[order], ends[order]
    k = np.searchsorted(a, sample_ids, side='right') - 1
    return (k >= 0) & (sample_ids <= b[np.clip(k, 0, None)])

def _velocity(deg, ts, circular=False, smooth=5):
    v = np.unwrap(np.deg2rad(deg)) if circular else np.deg2rad(deg)
    v = pd.Series(v).rolling(smooth, center=True, min_periods=1).mean().values
    return np.rad2deg(np.gradient(v, ts))


def drift_corrected_yaw(ori, mag, mag_A, mag_b, usable, window_s=10, min_resultant=.8):
    """yaw_imu minus its drift against magnetic north (deg, 0 = north), and the drift itself.

    ori: orientation table, mag: magnetometer table (Gauss), mag_A/mag_b: soft/hard-iron calibration (uT),
    usable: bool per orientation sample, False where the field estimate should not be used.
    """
    t_ori = ori.orientation_ephys_timestamp.values / 1e6
    t_mag = mag.magnetometer_ephys_timestamp.values / 1e6
    q6 = ori[[f'orientation_q_imu_{c}' for c in 'wxyz']].values
    M = (mag[[f'magnetometer_{c}' for c in 'xyz']].values * 100 - mag_b) @ mag_A.T     # Gauss -> uT, calibrated
    M = M * np.array([1., -1, 1])                                                       # magnetometer y is flipped vs the IMU
    oi = np.clip(np.searchsorted(t_ori, t_mag), 0, len(t_ori) - 1)
    ok = usable[oi]
    mw = _rotate(q6[oi[ok]], M[ok])
    north = np.arctan2(mw[:, 1], mw[:, 0])                     # north in the 6-axis world frame = - yaw drift
    grp = pd.Series(np.exp(1j * north)).groupby((t_mag[ok] // window_s).astype(int)).mean()
    grp = grp[grp.abs() > min_resultant]                       # skip windows with a disturbed field
    t_win = (grp.index.values + .5) * window_s
    drift = np.rad2deg(np.interp(t_ori, t_win, np.unwrap(np.angle(grp.values))))
    return _wrap(ori.orientation_yaw_imu.values - drift), drift


def get_head_orientation(session_fullfname, ori, mag, problems, cam2_offset):
    """HeadOrientation analytic (see module docstring). ori, mag, problems: the session's orientation, magnetometer and
    problem_intervals modalities."""
    L = Logger()
    sample_id = ori.orientation_sample_id.values
    pads = problems[problems.category.str.startswith('zero_pad')]
    malf = problems[problems.category == 'logger_malfunction_window']
    near_pad = _in_intervals(sample_id, pads.start_sample.values - 2000, pads.end_sample.values + 2000)   # +-100 ms
    in_malf = _in_intervals(sample_id, malf.start_sample.values, malf.end_sample.values)

    calib = os.path.join(os.path.dirname(session_fullfname), '_original', 'orientation_rpy.mat')
    if os.path.exists(calib):
        with h5py.File(calib, 'r') as f:
            mag_A, mag_b = f['A'][()], f['b'][()].ravel()
    else:
        L.logger.warning(f"No magnetometer calibration at {calib}, using the uncalibrated field")
        mag_A, mag_b = np.eye(3), np.zeros(3)
    yaw_magref, drift = drift_corrected_yaw(ori, mag, mag_A, mag_b, ~near_pad & ~in_malf)

    t = ori.orientation_ephys_timestamp.values / 1e6
    head_direction = _wrap(yaw_magref + cam2_offset)
    w, x, y, z = (ori[f'orientation_q_imu_{c}'].values for c in 'wxyz')
    # world-z components of the logger axes: x (left ear) 2(xz - wy), y (towards the tail) 2(yz + wx); snout = -y
    snout_elevation = -np.degrees(np.arcsin(np.clip(2 * (y * z + w * x), -1, 1)))
    head_tilt = np.degrees(np.arcsin(np.clip(2 * (x * z - w * y), -1, 1)))
    return pd.DataFrame({
        'sample_id': sample_id,
        'ephys_timestamp': ori.orientation_ephys_timestamp.values,
        'head_direction': head_direction,
        'head_direction_vel': _velocity(head_direction, t, circular=True),
        'snout_elevation': snout_elevation,
        'snout_elevation_vel': _velocity(snout_elevation, t),
        'head_tilt': head_tilt,
        'head_tilt_vel': _velocity(head_tilt, t),
        'yaw_drift': drift,
        'ephys_patched': ori.orientation_ephys_patched.values.astype(bool),
        'near_zero_pad': near_pad,
        'in_malfunction_window': in_malf,
        'clean': ~near_pad & ~in_malf,
    })
