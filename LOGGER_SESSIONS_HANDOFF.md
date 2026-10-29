# Logger arena sessions: handoff

Free-moving rat with a head-mounted wireless logger (Intan512 ephys + IMU) on a round table with a walking robot, filmed by
6 synchronized cameras. First (and so far only) session: `2026-09-09_15-15_rPG007_P1500_RobotInteract_45min`.
This file is the starting point for picking up the work; details live in the module docstrings named below.

## Where things are

| what | where |
|---|---|
| session (NAS) | `<nas>/RUN_rPG007/rPG007_P1500/2026-09-09_15-15_rPG007_P1500_RobotInteract_45min/`: `<session>.hdf5` (all modalities), `README.md` (session layout, timebase, problem intervals), `rendered_videos/cam1-6.mp4` (+ `cam2_arena/`, the annotated video), `session_analytics/*.parquet`, `_original/` (raw extractor output, read-only; the magnetometer calibration `orientation_rpy.mat` is read from here) |
| NAS root | macOS `/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/`, Linux `/mnt/SpatialSequenceLearning/` (`baseVR.base_functionality.device_paths`) |
| DLC project | `<nas>/dlc_pose_estim/ratArenaLogger-Simon-2026-10-28/` (shuffles 1-6; 6 is in use) |
| notebooks | `scripted_plotting/logger_session_analysis.ipynb` (start here: loads analytics, plots, tuning), `scripted_plotting/logger_session_overview.ipynb` (long data-reliability audit of the raw modalities; static, its builder is not in the repo) |
| video | `scripted_plotting/render_logger_arena_video.py` -> `<session>/rendered_videos/cam2_arena/cam2_arena_*min.mp4` (NAS) |

## Machines and environments

- **analysisVR** conda env: everything except DLC. Run scripts from `analysisVR/` (`python run_pipeline.py ...`).
- **DEEPLABCUT** conda env on the Linux GPU PC (`vrmaster`, RTX 3080, reached by SSH): DLC training and `BehaviorPose`.
  On the Mac, DLC inference segfaults (MPS and CPU, torch 2.2.2) - don't try. Labelling on the Mac works.
- `device_paths()` keys on `os.getlogin()` (macOS: `root`, Linux: `vrmaster`/`houmanjava`). In shells without a login tty
  (nbconvert, some agent shells) `os.getlogin()` fails; patch it in a test harness (`os.getlogin = lambda: "root"`), not in the repo.

## Pipeline (per session)

```bash
# DEEPLABCUT env, GPU PC
python run_pipeline.py BehaviorPose --session_names <session>            # DLC on cam2, ~1 h; raw keypoints + timestamps
# analysisVR env, any machine (seconds)
python run_pipeline.py HeadOrientation --session_names <session>         # IMU head angles
python run_pipeline.py ArenaPoseFeatures --session_names <session>       # needs BehaviorPose + HeadOrientation
python run_pipeline.py Spikes-SpikeClusterMetadata --session_names <session>
python run_pipeline.py LoggerMalfunction --session_names <session>       # needs Spikes-SpikeClusterMetadata
python scripted_plotting/render_logger_arena_video.py <session>          # video, ~15 min, sequential (one ffmpeg at a time, <1 GB RAM)
```
Add `--mode recompute` to overwrite. Keep the order `Spikes-SpikeClusterMetadata`: combo analytics are saved by position, the
reversed name would swap the two files. Code: `analytics_processing/analytics.py` (dispatch), `agg_modalities2analytic.py`
(`get_BehaviorPose`, `get_HeadOrientation`, `get_ArenaPoseFeatures`), `general_processing/logger_head_orientation.py`,
`general_processing/arena_pose_features.py`, `ephys_preprocessing/postproc_logger_ephys.py` (`is_logger_session`, logger spikes).

## Analytics and their columns

- **HeadOrientation** (100 Hz, `logger_head_orientation.py`): `head_direction` (snout, cam2 frame), `snout_elevation`, `head_tilt`,
  `*_vel`, `yaw_drift`, `clean` (not within 100 ms of a data-loss pad, not in the malfunction window).
- **BehaviorPose** (30 Hz, cam2): raw DLC `<kp>_x/_y/_likelihood`, `image_id`, `image_ephys_timestamp`, `image_in_malfunction_window`,
  `dlc_scorer` (model, shuffle, snapshot).
- **ArenaPoseFeatures** (30 Hz, `arena_pose_features.py`): the above plus `<kp>_x_filt/_y_filt/_filt_std` (Kalman/RTS smoothed, short gaps
  filled, a validation gate drops detections that jump to another object), `head_x/_y` (logger LED), `movement_energy`, robot centre
  `robot_x/_y` with `robot_n_points` (2 measured, 1 inferred from one point, 0 none) and `robot_inferred_age_s`, IMU angles at the frame
  times, `robot_distance`, `robot_bearing` (IMU; 0 = robot ahead), `robot_bearing_dlc`, `turn_towards_robot`, `head_direction_dlc`.
- **Spikes / SpikeClusterMetadata**: curated JRClust `single` units only (186). Spikes has the MEA1K columns, order and dtypes, without the
  waveform columns (`*_hpf_wf`, not extracted); `channel_3rd` is -1 (this JRClust run has no third site). No reliability flags on the spikes.
- **LoggerMalfunction** (`postproc_logger_ephys.get_logger_malfunction`): unreliable periods, one row each: `kind`, `category`,
  `start/end_sample`, `start/end_ephys_timestamp`, `duration_s`. Kinds: `zero_pad` (data replaced by zeros, 53), `zero_pad_margin`
  (pad +-30 ms, spike detections disturbed around the steps), `logger_window` (2268.7-2292.8 s), `malfunction_period` (the window
  grown over the population-rate burst around it, 2256-2337 s). For spikes exclude `zero_pad_margin` and `malfunction_period`:
  `postproc_logger_ephys.in_logger_malfunction(spikes.ephys_timestamp, malfunction)` (4.2 % of spikes).

## Conventions (easy to get wrong)

- **IMU mounting**: logger x = left ear, y = towards the tail (snout = -y), z = crown. So `snout_elevation` = -`roll_imu` (exact form from the
  quaternion), `head_tilt` = -`pitch_imu`, and `yaw_imu` is the direction of the left ear. Found from the data and video frames at the
  angle extremes; an earlier assumption (x = snout) was wrong.
- Use the 6-axis fusion (`orientation_*_imu`, `orientation_q_imu_*`); never the 9-axis `orientation_yaw/pitch/roll` (magnetometer y axis
  is flipped). The 6-axis yaw drifts ~25 deg/min; `HeadOrientation` removes it with the magnetometer (10 s windows).
- **Angles** are degrees in [-180, 180), counterclockwise as seen from above, in the **cam2 image frame**: 0 = image right, 90 = away
  from cam2. `LOGGER_CAM2_HEADING_OFFSET = -45` maps the north-referenced left-ear yaw to the snout direction in this frame (fitted from
  the walking direction; it absorbs the 90 deg mounting). Checked: DLC ear-line heading = IMU head direction +- 10 deg (median), no trend.
- **Positions** are cam2 image px; distances/speeds are "floor px" (image y / `DLC_ARENA_FLOOR_Y_SCALE` = 0.69). The floor ellipse
  (fitted from the video, b/a = 0.693) agrees with that scale. No metric units yet.
- **Time**: every table has `ephys_timestamp` (us, ephys clock = sample_id * 50). Ephys time has spurious 192 ms zero pads, so it runs
  ahead of real time; all modalities share it, so they align. Camera row k = mp4 frame k for all 6 cameras.
- Session-specific constants: `LOGGER_CAM2_HEADING_OFFSET`, `DLC_ARENA_FLOOR_Y_SCALE`, the burst window. Re-check them for a new session
  (another camera placement or logger mounting changes them).

## DLC (`general_processing/dlc_arena.py`, run in the DEEPLABCUT env from `analysisVR/`)

```bash
python -m general_processing.dlc_arena add <session.hdf5> --n 60          # pick frames: one k-means over all 6 cameras
python -m general_processing.dlc_arena add <session.hdf5> --more 30       # more frames, labels kept
python -m general_processing.dlc_arena label [--cam cam3]                 # napari; Ctrl+S, close, repeat
python -m general_processing.dlc_arena train [--epochs 1000]              # new shuffle each run
python -m general_processing.dlc_arena train --init-from 6 --epochs 300   # continue from a shuffle's final snapshot
```
Then set `DLC_ARENA_SHUFFLE` in `analytics_constants.py`. Lessons: with ~60 frames DLC's default 200 epochs (LR drops at 90/120)
never fit (train RMSE ~150 px); 1000 epochs with drops at 75/90 % works (shuffle 4: train RMSE 28 px). DLC's "best" snapshot is
picked on 3 test frames, so `DLC_ARENA_SNAPSHOT_INDEX = -2` uses the final one. `DLC_ARENA_BATCH_SIZE = 8` (full frames, 10 GB GPU).
Gotchas: macOS creates XSym files instead of symlinks on the SMB share (Linux sees plain files), so `add` reads the session video
directly; pandas 3 breaks napari-deeplabcut's reader (`label()` switches `future.infer_string` off).

## Status and next steps

- Tracking (shuffle 6, cam2): LED 93 % tracked, ears 75/49 %, robot back 83 %, robot front 37 % (hidden when the robot faces away),
  nose 19 %. Robot centre known in 97 % of frames (56 % from both points). See notebook section 3.
- Behaviour: the rat mostly faces away from the robot (robot within 30 deg of the head direction in ~8 % of frames); 12 turns towards it,
  followed by approach. Tuning (single units, shuffle p99): 75/186 head direction, 108/186 snout elevation - both confounded with
  position/behaviour, not yet controlled.
- **3D from all six cameras** (planned, not started): calibrate the cameras (best: a ChArUco recording + `aniposelib`; otherwise bundle
  adjustment on the LED seen by all cameras, arena rim for scale), run `BehaviorPose` on all cameras, triangulate per keypoint with
  outlier rejection, smooth in 3D with rigid-body constraints (robot on the floor: position + yaw), arena coordinates in mm.
- Arena coordinates / metric units, a homography for cam2, and controls for the tuning (position, speed) are open.
