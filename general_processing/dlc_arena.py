"""DLC project for the logger arena sessions (free-moving rat + robot). One model is trained on frames from all six arena
cameras (each sees the rat from another side, e.g. both ears); sessions are analyzed on cam2 (DLC_ARENA_CAMERA_FNAME).

Run in the DEEPLABCUT conda env. Steps (--cam takes a camera, a comma separated list, or 'all'):
    1. create the project once:          python -m general_processing.dlc_arena create <session.hdf5>
    2. add a session + pick frames:      python -m general_processing.dlc_arena add <session.hdf5> [--n 60] [--cam all]
       (one k-means over all cameras' thumbnails picks n frames in total; see add_session)
    3. label (DLC napari GUI):           python -m general_processing.dlc_arena label [--cam cam3]
       (opens one camera's frames at a time, by default the first without saved labels; run again for the next one)
    4. train:                            python -m general_processing.dlc_arena train [--epochs 1000]
       (each run is a new shuffle, earlier models are kept; then set DLC_ARENA_SHUFFLE in analytics_constants.py)
    more frames later, keeping the labeled ones:
       python -m general_processing.dlc_arena add <session.hdf5> --more 30
       python -m general_processing.dlc_arena label                (opens the next folder with unlabeled frames)
       python -m general_processing.dlc_arena train --init-from 4 --epochs 300   (new shuffle, starts from shuffle 4's final snapshot)
    5. analyze sessions:                 get_analytics('BehaviorPose', ...) -> agg_modalities2analytic.get_BehaviorPose

Session videos are linked into <project>/videos as <session>_<cam>.mp4 with a *relative* symlink, so the link works on every
machine that mounts the NAS share (macOS /Volumes/..., Linux /mnt/...). Hard links (link_dlc_training_data) are not
supported on the macOS SMB mount.
"""
import argparse
import os
import sys

if __name__ == '__main__':
    # executable - set up paths for import (run from analysisVR/)
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from baseVR.base_functionality import init_import_paths
    init_import_paths()

import cv2
import numpy as np
import pandas as pd
import yaml

from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as aC

ARENA_CAMS = [f'cam{i}' for i in range(1, 7)]
BODYPARTS = ['nose', 'left_ear', 'right_ear', 'logger_led', 'logger_back', 'tail_base', 'robot_front', 'robot_back']
SKELETON = [['nose', 'left_ear'], ['nose', 'right_ear'], ['left_ear', 'right_ear'], ['logger_back', 'logger_led'],
            ['logger_back', 'tail_base'], ['robot_back', 'robot_front']]


def project_dir():
    return os.path.join(aC.device_paths()[0], aC.DLC_ARENA_MODEL_SUBDIR)


def create_project(session_fullfname, cam=None, experimenter='Simon'):
    """Creates the DLC project with the arena bodyparts. The project name/date must match DLC_ARENA_MODEL_SUBDIR.
    DLC refuses to create a project without a video, so it gets this session's video; its absolute link is then replaced by
    add_session's relative one."""
    import shutil
    import deeplabcut
    L = Logger()
    pdir = project_dir()
    config = os.path.join(pdir, 'config.yaml')
    if os.path.exists(config):
        L.logger.info(f"Project exists: {pdir}")
        return config
    cam = cam or aC.DLC_ARENA_CAMERA_FNAME.replace('.mp4', '')
    video = os.path.join(os.path.dirname(session_fullfname), 'rendered_videos', f'{cam}.mp4')
    task, scorer, _ = os.path.basename(pdir).split('-', 2)
    deeplabcut.create_new_project(task, scorer, [video], working_directory=os.path.dirname(pdir), copy_videos=False)
    if not os.path.exists(config):
        raise RuntimeError(f"DLC did not create {pdir} (the date in DLC_ARENA_MODEL_SUBDIR must be today)")
    if os.path.islink(os.path.join(pdir, 'videos', f'{cam}.mp4')):
        os.remove(os.path.join(pdir, 'videos', f'{cam}.mp4'))
    shutil.rmtree(os.path.join(pdir, 'labeled-data', cam), ignore_errors=True)
    cfg = _read(config)
    cfg.update(bodyparts=BODYPARTS, skeleton=SKELETON, numframes2pick=60, dotsize=6, video_sets={})
    _write(config, cfg)
    L.logger.info(f"Created {config}")
    return config


def add_session(session_fullfname, cams=None, n_frames=60, step=10, min_gap_s=2.0, replace=False, more=0):
    """Links the session's camera videos into the project and writes n_frames frames to label, pooled over the cameras.

    Frames are picked like DLC's kmeans extraction (one frame per cluster of small grayscale thumbnails of every `step`-th
    frame), but with a single k-means over the thumbnails of all cameras together, so the n_frames follow the joint
    distribution (viewpoints, rat and robot positions) instead of a fixed number per camera. Picks are at least `min_gap_s`
    apart in time, also across cameras, since the cameras film the same moments. Frames in the logger malfunction window are
    skipped. The thumbnails come from one sequential pass per video (seeking in the 8 GB h264 files is slow) and are cached in
    <project>/thumbnails, so picking again with another n_frames is quick. Each camera's frames go to
    labeled-data/<session>_<cam>; a camera with no picks gets no folder.
    replace: move this session's existing unlabeled frame folders to <project>/unused-frames first (labeled ones are kept).
    more: add this many frames to the existing ones (labels are kept). The k-means then has (existing + more) clusters and
        new frames come only from clusters without an existing frame, i.e. from what the current frames do not cover.
    """
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.cluster import MiniBatchKMeans
    L = Logger()
    cams = cams or ARENA_CAMS
    config = os.path.join(project_dir(), 'config.yaml')
    cfg = _read(config)
    session_name = os.path.basename(session_fullfname)[:-5]

    out_dirs = {cam: os.path.join(project_dir(), 'labeled-data', f'{session_name}_{cam}') for cam in cams}
    existing = [d for d in out_dirs.values() if os.path.isdir(d) and any(f.endswith('.png') for f in os.listdir(d))]
    labeled = [d for d in existing if os.path.exists(os.path.join(d, f"CollectedData_{cfg['scorer']}.h5"))]
    if existing and not replace and not more:
        L.logger.info(f"Frames already extracted for {session_name} ({len(existing)} folders), nothing to do. "
                      f"Use --more N to add N frames, or --replace to move the unlabeled ones to unused-frames/ and pick again.")
        return None
    if replace and more:
        raise ValueError("Use either replace or more")
    if replace and labeled:
        raise RuntimeError(f"Labels saved in {labeled} - not replacing labeled frames")
    old_picks = [(i, int(f[3:-4])) for i, cam in enumerate(cams) if os.path.isdir(out_dirs[cam])
                 for f in os.listdir(out_dirs[cam]) if f.startswith('img') and f.endswith('.png')] if more else []
    for d in (existing if replace else []):
        dest = os.path.join(project_dir(), 'unused-frames', f"{os.path.basename(d)}_{time.strftime('%Y%m%d-%H%M%S')}")
        L.logger.info(f"Moving {d} -> {dest}")
        shutil.move(d, dest)

    videos = {cam: os.path.join(os.path.dirname(session_fullfname), 'rendered_videos', f'{cam}.mp4') for cam in cams}
    for cam, video in videos.items():
        link = os.path.join(project_dir(), 'videos', f'{session_name}_{cam}.mp4')
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(video, os.path.dirname(link)), link)
        cap = cv2.VideoCapture(video)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        video_sets = cfg.get('video_sets') or {}
        if not any(os.path.basename(v) == os.path.basename(link) for v in video_sets):   # one entry per video, any machine
            cfg['video_sets'] = {**video_sets, link: {'crop': f'0, {w}, 0, {h}'}}
    _write(config, cfg)

    def thumbnails(cam):
        cache = os.path.join(project_dir(), 'thumbnails', f'{session_name}_{cam}_step{step}.npz')
        if os.path.exists(cache):
            d = np.load(cache)
            return d['X'], d['ids']
        # read the session video itself: a symlink made from the macOS SMB mount is stored as an XSym text file, which a
        # Linux CIFS mount without the mfsymlinks option shows as a plain (non-video) file
        cap = cv2.VideoCapture(videos[cam])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n == 0:
            raise RuntimeError(f"Could not read {videos[cam]}")
        L.logger.info(f"Thumbnails of every {step}th of {n:,} frames in {videos[cam]} ...")
        thumbs, ids = [], []
        for f in range(n):
            if not cap.grab():
                break
            if f % step:
                continue
            img = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY)
            thumbs.append(cv2.resize(img, (48, 30), interpolation=cv2.INTER_AREA).ravel())
            ids.append(f)
        cap.release()
        X, ids = np.asarray(thumbs, np.uint8), np.asarray(ids)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, X=X, ids=ids)
        return X, ids

    with ThreadPoolExecutor(len(cams)) as ex:                 # cv2 decodes outside the GIL
        thumbs = dict(zip(cams, ex.map(thumbnails, cams)))

    X, frame, cam_i = [], [], []
    for i, cam in enumerate(cams):
        Xc, ids = thumbs[cam]
        packages = pd.read_hdf(session_fullfname, f'{cam}_packages')
        bad = packages.get(f'{cam}_image_in_malfunction_window', pd.Series(False, index=packages.index)).values
        keep = ~bad[np.minimum(ids, len(bad) - 1)]
        X.append(Xc[keep]); frame.append(ids[keep]); cam_i.append(np.full(keep.sum(), i))
    X, frame, cam_i = np.concatenate(X).astype(np.float32) / 255, np.concatenate(frame), np.concatenate(cam_i)

    n_new = more or n_frames
    km = MiniBatchKMeans(n_clusters=len(old_picks) + n_new, n_init=3, random_state=0).fit(X)
    dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    row = {(c, f): j for j, (c, f) in enumerate(zip(cam_i, frame))}
    covered = {km.labels_[row[p]] for p in old_picks if p in row}
    fps = 30
    picks = list(old_picks)                                   # (cam index, frame), the member closest to each centroid
    for c in np.argsort(-np.bincount(km.labels_, minlength=km.n_clusters)):   # big clusters first
        if c in covered or len(picks) - len(old_picks) == n_new:
            continue
        for j in np.flatnonzero(km.labels_ == c)[np.argsort(dist[km.labels_ == c])]:
            if all(abs(frame[j] - f) >= min_gap_s * fps for _, f in picks):
                picks.append((cam_i[j], frame[j]))
                break
    picks = picks[len(old_picks):]
    if more:
        L.logger.info(f"{len(old_picks)} existing frames cover {len(covered)} of {km.n_clusters} clusters; adding {len(picks)}")

    for i, cam in enumerate(cams):
        frames = sorted(f for ci, f in picks if ci == i)
        if not frames:
            continue
        os.makedirs(out_dirs[cam], exist_ok=True)
        cap = cv2.VideoCapture(videos[cam])
        width = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))   # DLC's frame file naming: img + zero-padded frame index
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                cv2.imwrite(os.path.join(out_dirs[cam], f'img{f:0{width}d}.png'), img)
        cap.release()
        L.logger.info(f"Wrote {len(frames)} {'new ' if more else ''}frames to {out_dirs[cam]}")
    return picks


def label(cam=None):
    """Opens one labeled-data folder in napari: the given camera's, else the first one without saved labels."""
    import deeplabcut
    import napari
    if pd.__version__ >= '3':
        # napari-deeplabcut (0.2.1.7) cannot read saved labels with pandas 3's default string dtype
        pd.set_option('future.infer_string', False)
    L = Logger()
    config = os.path.join(project_dir(), 'config.yaml')
    labels_dir = os.path.join(project_dir(), 'labeled-data')
    folders = sorted(d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d)))
    if not folders:
        raise RuntimeError(f"No frames to label in {labels_dir} - run the 'add' step first (and let it finish)")
    todo = [d for d in folders if _unlabeled_images(os.path.join(labels_dir, d), _read(config)['scorer'])]
    if cam is not None:
        folders = [d for d in folders if d.endswith(f'_{cam}')]
        if not folders:
            raise ValueError(f"No labeled-data folder for {cam} - run the 'add' step with --cam {cam} first")
    folder = next((d for d in folders if d in todo), folders[0])
    L.logger.info(f"Labeling {folder}. Folders with unlabeled frames ({len(todo)}): {todo}")
    deeplabcut.label_frames(config, folder)
    napari.run()                      # DLC only creates the viewer (enough in IPython); a script exits without the event loop


def _unlabeled_images(folder, scorer):
    """PNGs in a labeled-data folder that are not in its CollectedData file yet."""
    pngs = {f for f in os.listdir(folder) if f.endswith('.png') and not f.startswith('._')}
    h5 = os.path.join(folder, f"CollectedData_{scorer}.h5")
    if not os.path.exists(h5):
        return sorted(pngs)
    labeled = {i[-1] if isinstance(i, tuple) else os.path.basename(i) for i in pd.read_hdf(h5).index}
    return sorted(pngs - labeled)


def existing_shuffles():
    """Shuffle indices that already have a training dataset or a model folder."""
    import glob
    import re
    pdir = project_dir()
    paths = glob.glob(os.path.join(pdir, 'training-datasets', 'iteration-*', '*', 'Documentation_data-*shuffle*.pickle')) + \
            glob.glob(os.path.join(pdir, 'dlc-models*', 'iteration-*', '*shuffle*'))
    return sorted({int(re.search(r'shuffle(\d+)', os.path.basename(p)).group(1)) for p in paths})


def train(shuffle=None, net_type='resnet_50', epochs=1000, init_from=None):
    """Creates a training dataset from all saved labels and trains it as a new shuffle (default: the next free index), so
    earlier models are kept. Set aC.DLC_ARENA_SHUFFLE to the new index to analyze sessions with it.
    init_from: shuffle index whose final snapshot the new shuffle starts from (e.g. after labeling more frames); training
        then continues at that snapshot's epoch for `epochs` more epochs."""
    import deeplabcut
    L = Logger()
    config = os.path.join(project_dir(), 'config.yaml')
    # without labels DLC's create_training_dataset only prints "No data was found!" and train_network then fails with a
    # misleading "Could not find a shuffle"
    labels_dir = os.path.join(project_dir(), 'labeled-data')
    labels = [d for d in os.listdir(labels_dir) if os.path.exists(os.path.join(labels_dir, d, f"CollectedData_{_read(config)['scorer']}.h5"))]
    if not labels:
        raise RuntimeError(f"No labeled frames in {labels_dir} - run the 'label' step first and save in the GUI")
    existing = existing_shuffles()
    if shuffle is None:
        shuffle = max(existing, default=0) + 1
    elif shuffle in existing:
        raise ValueError(f"Shuffle {shuffle} exists already (existing: {existing}), pick another one or leave --shuffle out")
    snapshot_path, start = None, 0
    if init_from is not None:
        import glob
        import re
        snaps = glob.glob(os.path.join(project_dir(), 'dlc-models-pytorch', 'iteration-*', f'*shuffle{init_from}', 'train',
                                       'snapshot-[0-9]*.pt'))
        if not snaps:
            raise ValueError(f"No snapshots for shuffle {init_from}")
        snapshot_path = max(snaps, key=lambda p: int(re.search(r'snapshot-(\d+)', p).group(1)))   # final epoch, not 'best'
        start = int(re.search(r'snapshot-(\d+)', snapshot_path).group(1))
        L.logger.info(f"Starting from {snapshot_path} (epoch {start})")
    L.logger.info(f"Training shuffle {shuffle} on {len(labels)} labeled folders: {labels}")
    deeplabcut.create_training_dataset(config, Shuffles=[shuffle], net_type=net_type)
    # DLC's default schedule drops the learning rate at epochs 90 and 120. With ~60 frames (8 steps per epoch at batch 8)
    # that is only ~1000 steps at a useful learning rate, and the model never fits (train RMSE ~150 px, likelihoods < 0.6).
    # Train longer and move the drops to 75 % and 90 % of the epochs (counted from `start` when continuing a snapshot).
    # A snapshot stores its scheduler (already past its last drop, lr 1e-5); not loading it restarts at the base lr.
    cfg_updates = {"runner.scheduler.params.milestones": [start + int(epochs * .75), start + int(epochs * .9)]}
    if snapshot_path:
        cfg_updates["runner.load_scheduler_state_dict"] = False
    deeplabcut.train_network(config, shuffle=shuffle, epochs=epochs, save_epochs=max(epochs // 10, 1), snapshot_path=snapshot_path,
                             pytorch_cfg_updates=cfg_updates)
    deeplabcut.evaluate_network(config, Shuffles=[shuffle], plotting=True)
    L.logger.info(f"Done. To analyze sessions with this model set DLC_ARENA_SHUFFLE = {shuffle} in analytics_constants.py")


def _read(config):
    with open(config) as f:
        return yaml.safe_load(f)

def _write(config, cfg):
    with open(config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['create', 'add', 'label', 'train'])
    ap.add_argument('session_fullfname', nargs='?')
    ap.add_argument('--cam', default=None)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--shuffle', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--more', type=int, default=0)
    ap.add_argument('--init-from', type=int, default=None)
    args = ap.parse_args()
    Logger().init_logger(None, None, 'INFO')
    if args.step == 'create':
        create_project(args.session_fullfname, args.cam)
    elif args.step == 'add':
        cams = ARENA_CAMS if args.cam in (None, 'all') else args.cam.split(',')
        add_session(args.session_fullfname, cams, args.n, replace=args.replace, more=args.more)
    elif args.step == 'label':
        label(args.cam)
    else:
        train(args.shuffle, epochs=args.epochs, init_from=args.init_from)
