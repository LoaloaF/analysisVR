"""DLC project for the logger arena sessions (free-moving rat + robot, single camera, default cam2).

Run in the DEEPLABCUT conda env. Steps:
    1. create the project once:          python -m general_processing.dlc_arena create <session.hdf5>
    2. add a session + pick frames:      python -m general_processing.dlc_arena add <session.hdf5> [--n 150]
    3. label (DLC GUI):                  deeplabcut.label_frames(<config.yaml>)
    4. train:                            python -m general_processing.dlc_arena train
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
    cfg.update(bodyparts=BODYPARTS, skeleton=SKELETON, numframes2pick=150, dotsize=6, video_sets={})
    _write(config, cfg)
    L.logger.info(f"Created {config}")
    return config


def add_session(session_fullfname, cam=None, n_frames=150, step=10):
    """Links the session's camera video into the project and writes n_frames frames to label.

    Frames are picked like DLC's kmeans extraction (one frame per cluster of small grayscale thumbnails, every `step`-th frame),
    but from one sequential pass over the video (seeking in the 8 GB h264 files is slow) and without frames in the logger
    malfunction window.
    """
    from sklearn.cluster import MiniBatchKMeans
    L = Logger()
    cam = cam or aC.DLC_ARENA_CAMERA_FNAME.replace('.mp4', '')
    config = os.path.join(project_dir(), 'config.yaml')
    cfg = _read(config)

    session_name = os.path.basename(session_fullfname)[:-5]
    video = os.path.join(os.path.dirname(session_fullfname), 'rendered_videos', f'{cam}.mp4')
    link = os.path.join(project_dir(), 'videos', f'{session_name}_{cam}.mp4')
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(video, os.path.dirname(link)), link)
    cap = cv2.VideoCapture(link)
    n, w, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfg['video_sets'] = {**(cfg.get('video_sets') or {}), link: {'crop': f'0, {w}, 0, {h}'}}
    _write(config, cfg)

    packages = pd.read_hdf(session_fullfname, f'{cam}_packages')
    bad = packages.get(f'{cam}_image_in_malfunction_window', pd.Series(False, index=packages.index)).values
    L.logger.info(f"Thumbnails of every {step}th of {n:,} frames in {link} ...")
    thumbs, ids = [], []
    for f in range(n):
        if not cap.grab():
            break
        if f % step or (f < len(bad) and bad[f]):
            continue
        img = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY)
        thumbs.append(cv2.resize(img, (48, 30), interpolation=cv2.INTER_AREA).ravel())
        ids.append(f)
    X = np.asarray(thumbs, np.float32) / 255
    km = MiniBatchKMeans(n_clusters=n_frames, n_init=3, random_state=0).fit(X)
    dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    picks = sorted(ids[np.flatnonzero(km.labels_ == c)[np.argmin(dist[km.labels_ == c])]]
                   for c in range(n_frames) if (km.labels_ == c).any())

    out_dir = os.path.join(project_dir(), 'labeled-data', os.path.splitext(os.path.basename(link))[0])
    os.makedirs(out_dir, exist_ok=True)
    width = len(str(n))                                   # DLC's frame file naming: img + zero-padded frame index
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(os.path.join(out_dir, f'img{f:0{width}d}.png'), img)
    cap.release()
    L.logger.info(f"Wrote {len(picks)} frames to {out_dir}")
    return picks


def train(shuffle=1, net_type='resnet_50', epochs=200):
    import deeplabcut
    config = os.path.join(project_dir(), 'config.yaml')
    deeplabcut.create_training_dataset(config, Shuffles=[shuffle], net_type=net_type)
    deeplabcut.train_network(config, shuffle=shuffle, epochs=epochs)
    deeplabcut.evaluate_network(config, Shuffles=[shuffle], plotting=True)


def _read(config):
    with open(config) as f:
        return yaml.safe_load(f)

def _write(config, cfg):
    with open(config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['create', 'add', 'train'])
    ap.add_argument('session_fullfname', nargs='?')
    ap.add_argument('--cam', default=None)
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()
    Logger().init_logger(None, None, 'INFO')
    if args.step == 'create':
        create_project(args.session_fullfname, args.cam)
    elif args.step == 'add':
        add_session(args.session_fullfname, args.cam, args.n)
    else:
        train()
