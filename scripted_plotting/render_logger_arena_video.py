# !/usr/bin/env python3
"""Full-session cam2 video of a logger arena session, annotated from the ArenaPoseFeatures analytic.

    python scripted_plotting/render_logger_arena_video.py <session_name> [--chunks 0 1 2] [--out DIR]   (from analysisVR/)

One mp4 per 10 minutes, written next to the camera videos (<session>/rendered_videos/cam2_arena/), rendered one after another (~1-3 min per chunk, depending on how fast the NAS delivers the video);
a chunk is written as <name>.part.mp4 and renamed when complete. Layout (1920 x 1080):
- left: cam2 with the head (logger LED, ring = gap filled by the smoother), the IMU head direction (white arrow) and the
  robot centre (filled: both robot points tracked, ring: inferred from one point, pale ring: last two-point frame > 10 s away)
- right: the arena from above as seen from cam2 (the floor ellipse fitted from the video), occupancy over 5 minutes:
  rat (LED) coloured by time (viridis), robot grey, turns towards the robot as red rings + lines
- bottom: dials for the robot bearing (0 = robot ahead), the IMU head direction and the snout elevation (right half only)
  as polar timelines: the angle is the IMU angle, the radius is time (dashed ring = now, inside = the past 5 s, outside =
  the next 5 s), with gaps where the IMU is not clean or the robot position is missing. Right of them the movement energy
  over +-10 s with the turns towards the robot as red lines, and above it slim bars where the LED or the robot centre is
  not detected
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

import cv2
import numpy as np
from matplotlib import colormaps

from CustomLogger import CustomLogger as Logger
from analytics_processing import analytics
import analytics_processing.analytics_constants as C
import analytics_processing.sessions_from_nas_parsing as sp

CHUNK_FRAMES, SLICE_FRAMES, FPS = 18000, 9000, 30          # 10 min per file, occupancy reset every 5 min
W, H, VS = 1920, 1080, 2 / 3                               # output size, video scale (1920x1200 -> 1280x800)
WHITE, BLACK, GREY, RED = (255, 255, 255), (0, 0, 0), (150, 150, 150), (40, 40, 230)
ROBOT_C, LED_C, ELEV_C = (255, 0, 255), (0, 255, 255), (120, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIRIDIS = (np.array([colormaps['viridis'](i)[:3] for i in np.linspace(0, 1, 256)])[:, ::-1] * 255).astype(int)


def load_frames(session_name):
    """Per-frame arrays from ArenaPoseFeatures (only display smoothing is done here)."""
    f = analytics.get_analytics('ArenaPoseFeatures', session_names=[session_name]).reset_index(drop=True)
    return {'t': f.image_ephys_timestamp.values / 1e6, 'hx': f.head_x.values, 'hy': f.head_y.values,
            'led_raw': f.logger_led_likelihood.values >= C.DLC_MIN_LIKELIHOOD,
            'cx': f.robot_x.values, 'cy': f.robot_y.values, 'n_points': f.robot_n_points.values,
            'age': np.nan_to_num(f.robot_inferred_age_s.values, nan=np.inf),
            'hd': f.head_direction.values, 'elev': f.snout_elevation.values, 'bear': f.robot_bearing.values,
            'energy': np.convolve(np.nan_to_num(f.movement_energy.values), np.ones(5) / 5, 'same'),
            'malf': f.image_in_malfunction_window.values.astype(bool), 'turn': f.turn_towards_robot.values.astype(bool),
            'imu_clean': f.imu_clean.values.astype(bool)}


def runs(mask):
    """(start, end) of the True runs in a bool array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def fit_arena_ellipse(video, cache):
    """Floor ellipse (cx, cy, w, h, angle) in the video, from the bright floor in a median background frame."""
    if os.path.exists(cache):
        return np.load(cache)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []
    for k in np.linspace(300, n - 1000, 40).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, im = cap.read()
        if ok:
            imgs.append(im)
    hsv = cv2.cvtColor(np.median(np.stack(imgs), 0).astype(np.uint8), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 48) & (hsv[..., 1] > 150) & (hsv[..., 2] > 170)).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)), cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    contour = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0, :]
    (cx, cy), (w, h), ang = cv2.fitEllipse(contour[contour[:, 1] < m.shape[0] - 10])   # drop the image bottom edge
    ell = np.array([cx, cy, w, h, ang]); np.save(cache, ell)
    return ell


def render_chunk(d, ellipse, video, out_dir, chunk, n_frames=None):
    t, hx, hy, cx, cy = d['t'], d['hx'], d['hy'], d['cx'], d['cy']
    n_points, age, hd, elev, bear, energy, malf, is_turn = (d[k] for k in ['n_points', 'age', 'hd', 'elev', 'bear', 'energy', 'malf', 'turn'])
    led_raw, imu_ok = d['led_raw'], d['imu_clean']
    ys = C.DLC_ARENA_FLOOR_Y_SCALE
    f0 = chunk * CHUNK_FRAMES
    f1 = min(f0 + (n_frames or CHUNK_FRAMES), len(t))

    txt = lambda img, s, xy, scale=.5, col=WHITE, th=1: cv2.putText(img, s, (int(xy[0]), int(xy[1])), FONT, scale, col, th, cv2.LINE_AA)
    pt = lambda x, y: (int(round(x)), int(round(y)))
    window = lambda k, span: (np.searchsorted(t, t[k] - span), np.searchsorted(t, t[k] + span, side='right'))   # by time: ephys time jumps at pads

    # arena panel: top view as seen from cam2 (cam2 at the bottom)
    AX0, AY0, AW, AH, AR = 1280, 0, 640, 800, 285
    ACX, ACY = AX0 + AW // 2, AY0 + 430
    ecx, ecy, ew, eh, _ = ellipse
    ea, eb = max(ew, eh) / 2, min(ew, eh) / 2
    def to_arena(x, y):
        u, w = (x - ecx) / ea, (y - ecy) / eb
        r = max(np.hypot(u, w), 1e-9)
        if r > 1:                                                # head above the wall: pin to the rim
            u, w = u / r, w / r
        return ACX + u * AR, ACY + w * AR
    def arena_base(s0, s1):
        p = np.full((AH, AW, 3), 25, np.uint8)
        o = (ACX - AX0, ACY - AY0)
        cv2.circle(p, o, AR, (90, 90, 90), 2, cv2.LINE_AA)
        txt(p, 'arena from above (cam2 at the bottom)', (20, 34), .6)
        txt(p, f'occupancy {s0 / 60:.0f}-{s1 / 60:.0f} min', (20, 62), .55, (200, 200, 200))
        for i in range(0, 200, 2):
            cv2.rectangle(p, (20 + i, 80), (22 + i, 92), tuple(int(c) for c in VIRIDIS[int(i / 200 * 255)]), -1)
        txt(p, 'rat (LED): early', (20, 112), .45); txt(p, 'late', (190, 112), .45)
        cv2.circle(p, (330, 86), 5, GREY, -1); txt(p, 'robot', (342, 92), .45)
        cv2.circle(p, (420, 86), 9, RED, 2); txt(p, 'turn towards robot', (436, 92), .45)
        txt(p, 'cam2', (o[0] - 20, o[1] + AR + 30), .45, (160, 160, 160))
        txt(p, 'dials below: angle = IMU, dashed ring = now, inside = past 5 s, outside = next 5 s,', (20, 772), .42, (170, 170, 170))
        txt(p, 'gaps = IMU not clean (data loss, malfunction) or no robot position', (20, 790), .42, (170, 170, 170))
        return p

    # dials (polar timelines: angle = IMU angle, radius = time, dashed ring = now) and timeline
    DIAL_R, DIAL_Y, R_NOW, DIAL_SPAN = 88, 800 + 168, 50, 5.0
    DIALS = [(150, 'robot bearing (IMU)'), (420, 'head direction (IMU)'), (690, 'snout elevation (IMU)')]
    def dial_frame(img, cxd, title, labels, half=False):
        a0, a1 = (-90, 90) if half else (0, 360)                    # cv2 angles, clockwise from image right
        cv2.ellipse(img, (cxd, DIAL_Y), (DIAL_R, DIAL_R), 0, a0, a1, (110, 110, 110), 1, cv2.LINE_AA)
        if half:
            cv2.line(img, (cxd, DIAL_Y - DIAL_R), (cxd, DIAL_Y + DIAL_R), (70, 70, 70), 1)
        for a in range(a0, a1, 12):
            cv2.ellipse(img, (cxd, DIAL_Y), (R_NOW, R_NOW), 0, a, a + 6, (100, 100, 100), 1, cv2.LINE_AA)
        txt(img, title, (cxd - 95, 800 + 28), .5)
        for ang, lab in labels:
            a = np.deg2rad(ang)
            txt(img, lab, (cxd + (DIAL_R + 14) * np.cos(a) - 4 * len(lab), DIAL_Y - (DIAL_R + 14) * np.sin(a) + 5), .4, (170, 170, 170))
    def polar_trace(img, cxd, ang, ok, k, col):
        """ang (deg, 0 = right, ccw) +-DIAL_SPAN around frame k, radius = time; gaps where not ok. Returns whether k is ok."""
        lo, hi = window(k, DIAL_SPAN)
        r = R_NOW + (t[lo:hi] - t[k]) / DIAL_SPAN * (DIAL_R - R_NOW)
        a = np.deg2rad(ang[lo:hi])
        xy = np.stack([cxd + r * np.cos(a), DIAL_Y - r * np.sin(a)], 1)
        valid = ok[lo:hi] & ~np.isnan(a)
        now, dim = k - lo, tuple(int(c * .5) for c in col)
        for s, e in runs(valid):
            if s <= now:
                cv2.polylines(img, [xy[s:min(e, now + 1)].astype(np.int32)], False, col, 2, cv2.LINE_AA)
            if e > now:
                cv2.polylines(img, [xy[max(s, now):e].astype(np.int32)], False, dim, 1, cv2.LINE_AA)
        if valid[now]:
            cv2.circle(img, pt(*xy[now]), 6, col, -1, cv2.LINE_AA); cv2.circle(img, pt(*xy[now]), 6, BLACK, 1, cv2.LINE_AA)
        return valid[now]
    TL_X0, TL_X1, TL_Y0, TL_Y1, TL_SPAN, E_MAX = 860, 1890, 870, 1050, 10.0, 800.
    def timeline(img, k):
        cv2.rectangle(img, (TL_X0, TL_Y0), (TL_X1, TL_Y1), (60, 60, 60), 1)
        txt(img, 'movement energy (DLC, median keypoint speed, floor px/s)', (TL_X0, 800 + 30), .5)
        for e in (200, 400, 600):
            yy = TL_Y1 - e / E_MAX * (TL_Y1 - TL_Y0)
            cv2.line(img, (TL_X0, int(yy)), (TL_X1, int(yy)), (45, 45, 45), 1); txt(img, str(e), (TL_X0 - 34, yy + 4), .38, (150, 150, 150))
        lo, hi = window(k, TL_SPAN)
        xs = TL_X0 + (t[lo:hi] - t[k] + TL_SPAN) / (2 * TL_SPAN) * (TL_X1 - TL_X0)
        ysv = TL_Y1 - np.clip(energy[lo:hi], 0, E_MAX) / E_MAX * (TL_Y1 - TL_Y0)
        cv2.polylines(img, [np.stack([xs, ysv], 1).astype(np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)
        for y, lab, miss, col in ((TL_Y0 - 26, 'no LED', ~led_raw, LED_C), (TL_Y0 - 12, 'no robot', n_points == 0, ROBOT_C)):
            cv2.line(img, (TL_X0, y), (TL_X1, y), (55, 55, 55), 1)
            for s, e in runs(miss[lo:hi]):
                cv2.line(img, (int(xs[s]), y), (int(xs[e - 1]), y), col, 4)
            txt(img, lab, (TL_X0 - 62, y + 4), .38, (170, 170, 170))
        for j in np.flatnonzero(is_turn[lo:hi]):
            cv2.line(img, (int(xs[j]), TL_Y0), (int(xs[j]), TL_Y1), RED, 2)
        cv2.line(img, ((TL_X0 + TL_X1) // 2, TL_Y0), ((TL_X0 + TL_X1) // 2, TL_Y1), LED_C, 1)
        for dt in (-10, -5, 0, 5, 10):
            txt(img, f'{dt:+d} s' if dt else 'now', (TL_X0 + (dt + TL_SPAN) / (2 * TL_SPAN) * (TL_X1 - TL_X0) - 14, TL_Y1 + 20), .4, (170, 170, 170))

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f'cam2_arena_{f0 / FPS / 60:02.0f}-{f1 / FPS / 60:02.0f}min.mp4')
    part = fname[:-4] + '.part.mp4'                               # renamed once complete: a .mp4 name is always playable
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p', part],
                          stdin=subprocess.PIPE)
    t_start = time.time()
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    arena, slice_id = None, None
    for k in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        out = np.zeros((H, W, 3), np.uint8)
        v = cv2.resize(frame, (1280, 800), interpolation=cv2.INTER_AREA)
        if not np.isnan(hx[k]):
            p = pt(hx[k] * VS, hy[k] * VS)
            if not np.isnan(hd[k]):
                a = np.deg2rad(hd[k])
                cv2.arrowedLine(v, p, pt(p[0] + 70 * np.cos(a), p[1] - 70 * np.sin(a) * ys), WHITE, 3, cv2.LINE_AA, tipLength=.3)
            cv2.circle(v, p, 6, LED_C, -1 if d['led_raw'][k] else 2, cv2.LINE_AA)
        if n_points[k]:
            p = pt(cx[k] * VS, cy[k] * VS)
            if n_points[k] == 2:
                cv2.circle(v, p, 9, ROBOT_C, -1, cv2.LINE_AA)
            else:
                cv2.circle(v, p, 10, ROBOT_C if age[k] < 10 else (230, 170, 230), 3 if age[k] < 10 else 2, cv2.LINE_AA)
        cv2.rectangle(v, (0, 728), (640, 800), BLACK, -1)              # bottom left: table legs, never the rat
        mm, ss = divmod(t[k] - t[0], 60)
        txt(v, f'{int(mm):02d}:{ss:05.2f}  (ephys {t[k]:.1f} s, frame {k})', (10, 752), .6)
        txt(v, 'yellow: LED (ring = filled gap), white: IMU head direction', (10, 772), .45)
        txt(v, 'magenta: robot centre (filled: both points, ring: inferred, pale ring: > 10 s)', (10, 790), .45)
        if malf[k]:
            txt(v, 'logger malfunction window: IMU unreliable', (660, 790), .7, RED, 2)
        out[:800, :1280] = v

        if k // SLICE_FRAMES != slice_id:
            slice_id = k // SLICE_FRAMES
            arena = arena_base(slice_id * SLICE_FRAMES / FPS, min((slice_id + 1) * SLICE_FRAMES, len(t)) / FPS)
        col = tuple(int(c) for c in VIRIDIS[int((k - slice_id * SLICE_FRAMES) / SLICE_FRAMES * 255)])
        if n_points[k]:
            ax, ay = to_arena(cx[k], cy[k])
            cv2.circle(arena, pt(ax - AX0, ay - AY0), 2, GREY, -1)
        if not np.isnan(hx[k]):
            rx, ry = to_arena(hx[k], hy[k])
            cv2.circle(arena, pt(rx - AX0, ry - AY0), 2, col, -1)
        if is_turn[k] and n_points[k] and not np.isnan(hx[k]):
            cv2.line(arena, pt(rx - AX0, ry - AY0), pt(ax - AX0, ay - AY0), RED, 2, cv2.LINE_AA)
            cv2.circle(arena, pt(rx - AX0, ry - AY0), 10, RED, 2, cv2.LINE_AA)
        out[AY0:AY0 + AH, AX0:AX0 + AW] = arena
        if n_points[k]:
            cv2.circle(out, pt(ax, ay), 9, (200, 200, 200), -1, cv2.LINE_AA); cv2.circle(out, pt(ax, ay), 9, BLACK, 2, cv2.LINE_AA)
        if not np.isnan(hx[k]):
            cv2.circle(out, pt(rx, ry), 9, col, -1, cv2.LINE_AA); cv2.circle(out, pt(rx, ry), 9, WHITE, 2, cv2.LINE_AA)
            if not np.isnan(hd[k]):
                a = np.deg2rad(hd[k])
                cv2.arrowedLine(out, pt(rx, ry), pt(rx + 30 * np.cos(a), ry - 30 * np.sin(a)), WHITE, 2, cv2.LINE_AA, tipLength=.35)

        c0, c1, c2 = (x for x, _ in DIALS)
        dial_frame(out, c0, DIALS[0][1], [(90, '0'), (270, '180'), (180, 'L'), (0, 'R')])
        cv2.circle(out, (c0, DIAL_Y - DIAL_R), 7, GREY, -1)                 # the robot is "up" on this dial
        if is_turn[max(0, k - 30):k + 1].any():
            cv2.circle(out, (c0, DIAL_Y), DIAL_R + 4, RED, 3, cv2.LINE_AA)
        if polar_trace(out, c0, 90 + bear, imu_ok & ~np.isnan(bear), k, LED_C):
            label = 'robot ahead' if abs(bear[k]) < 30 else 'turned away' if abs(bear[k]) > 150 else ''
            txt(out, f'{abs(bear[k]):3.0f} deg {label}', (c0 - 95, 800 + 50), .45, LED_C)
        else:
            why = 'IMU data missing' if not imu_ok[k] else 'no head position (LED)' if np.isnan(hx[k]) else 'no robot position'
            txt(out, why, (c0 - 95, 800 + 50), .45, (150, 150, 150))
        dial_frame(out, c1, DIALS[1][1], [(0, 'R'), (90, 'away'), (180, 'L'), (270, 'cam2')])
        if polar_trace(out, c1, hd, imu_ok, k, WHITE):
            txt(out, f'{hd[k]:4.0f} deg', (c1 - 95, 800 + 50), .45)
        else:
            txt(out, 'IMU data missing', (c1 - 95, 800 + 50), .45, (150, 150, 150))
        dial_frame(out, c2, DIALS[2][1], [(0, '0'), (60, '+60'), (-60, '-60'), (90, 'up'), (-90, 'down')], half=True)
        cv2.line(out, (c2, DIAL_Y), (c2 + DIAL_R, DIAL_Y), (70, 70, 70), 1)
        if polar_trace(out, c2, elev, imu_ok, k, ELEV_C):
            txt(out, f'snout {elev[k]:+4.0f} deg', (c2 - 95, 800 + 50), .45, ELEV_C)
        else:
            txt(out, 'IMU data missing', (c2 - 95, 800 + 50), .45, (150, 150, 150))
        timeline(out, k)
        ff.stdin.write(out.tobytes())
        if (k - f0 + 1) % (60 * FPS) == 0:
            print(f'  {os.path.basename(fname)}: {(k - f0 + 1) / FPS / 60:.0f} of {(f1 - f0) / FPS / 60:.0f} min, '
                  f'{(k - f0 + 1) / (time.time() - t_start):.0f} frames/s', flush=True)
    ff.stdin.close(); ff.wait(); cap.release()
    os.replace(part, fname)
    return fname


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('session_name')
    ap.add_argument('--chunks', nargs='+', type=int, default=None, help='10-minute chunks to render (default: all)')
    ap.add_argument('--n_frames', type=int, default=None, help='frames per chunk (default: the whole chunk), for tests')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    Logger().init_logger(None, None, 'WARNING')
    session_fullfname = sp.sessionnames2fullfnames([args.session_name])[0][0]
    video = os.path.join(os.path.dirname(session_fullfname), 'rendered_videos', 'cam2.mp4')
    out_dir = args.out or os.path.join(os.path.dirname(video), 'cam2_arena')
    os.makedirs(out_dir, exist_ok=True)
    data = load_frames(args.session_name)
    ellipse = fit_arena_ellipse(video, os.path.join(out_dir, 'arena_ellipse_cam2.npy'))
    chunks = args.chunks if args.chunks is not None else list(range(int(np.ceil(len(data['t']) / CHUNK_FRAMES))))
    for chunk in chunks:
        print('wrote', render_chunk(data, ellipse, video, out_dir, chunk, args.n_frames), flush=True)
