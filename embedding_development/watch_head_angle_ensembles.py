#!/usr/bin/env python3
"""
watch_head_angle_ensembles.py — Video on left, head_angle + top correlated
ensemble activations on right.

Alignment: uses frame_pc_timestamp to find each frame's position in the
session-level sorted order — that position is the video frame index.
No manual trial_start offset needed.

Usage
-----
    python watch_head_angle_ensembles.py
    python watch_head_angle_ensembles.py --top_k 3 --trial_id 170
    python watch_head_angle_ensembles.py --save --save_path out.mp4

Controls
--------
    Play / Pause  — button or spacebar
    ×N button     — cycles speed (0.25×, 0.5×, 1×, 2×, 4×)
    Slider        — scrub
"""

import argparse
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import blended_transform_factory

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--trial_id",    type=int,   default=143)
parser.add_argument("--session",     type=str,   default="2025-01-26_21-48")
parser.add_argument("--video",       type=str,   default="./bodycam.mp4")
parser.add_argument("--base",        type=str,   default="./outputs/glm_input_data")
parser.add_argument("--top_k",       type=int,   default=4,
                    help="Number of top head_angle-attributed ensembles to show (default: 4)")
parser.add_argument("--attr",        type=str,   default="ig", choices=["ig", "gpv"],
                    help="Attribution method used to rank ensembles: 'ig' or 'gpv' (default: ig)")
parser.add_argument("--eval_dir",    type=str,
                    default="./outputs/mlps/ensembles_multiseed",
                    help="Directory containing importance_ig_semantic.npy / importance_global_pv_semantic.npy")
parser.add_argument("--speed",       type=float, default=1.0)
parser.add_argument("--window",      type=float, default=6.0,
                    help="Visible time window in seconds (default: 6)")
parser.add_argument("--save",        action="store_true",
                    help="Render to MP4 instead of interactive window")
parser.add_argument("--save_path",   type=str,   default=None)
parser.add_argument("--save_fps",    type=float, default=None)
parser.add_argument("--save_dpi",    type=int,   default=120)
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
base = args.base

beh = pd.DataFrame(
    np.load(f"{base}/behavior_glm_input.npy", allow_pickle=True),
    index=pd.Index(np.load(f"{base}/behavior_glm_input_index.npy", allow_pickle=True)),
    columns=np.load(f"{base}/behavior_glm_input_columns.npy", allow_pickle=True))

spk = pd.DataFrame(
    np.load(f"{base}/fr_full.npy"),
    index=pd.Index(np.load(f"{base}/fr_full_index.npy", allow_pickle=True)),
    columns=np.load(f"{base}/fr_full_columns.npy", allow_pickle=True))
spk.index = pd.MultiIndex.from_tuples(
    spk.index.map(lambda t: (t[0], t[1] // 40000 - 1)))

ensembles = np.load(f"{base}/ensembles.npy")   # (n_neurons, n_ensembles)
n_ensembles = ensembles.shape[1]

# ── Rank ensembles by IG / GPV attribution for head_angle ────────────────────
import pickle

attr_file = (f"{args.eval_dir}/importance_ig_semantic.npy" if args.attr == "ig"
             else f"{args.eval_dir}/importance_global_pv_semantic.npy")
attr_all = np.load(attr_file)                                      # (sessions, ensembles, groups)
sg       = pickle.load(open(f"{args.eval_dir}/semantic_groups.pkl", "rb"))
ha_group_idx = [g[0] for g in sg].index("head_angle")             # semantic group index

# Map session string → integer index (position in sorted unique sessions)
all_sessions = sorted(beh.index.map(lambda t: t[0]).unique())
if args.session not in all_sessions:
    sys.exit(f"Session '{args.session}' not found in data.")
sess_idx = all_sessions.index(args.session)

ha_attr = attr_all[sess_idx, :, ha_group_idx]                     # (n_ensembles,)
# NaN ensembles (dead/unused) get rank last
ha_attr_safe = np.where(np.isnan(ha_attr), -np.inf, ha_attr)
top_ensemble_idx = np.argsort(ha_attr_safe)[::-1][: args.top_k]

print(f"\nTop {args.top_k} ensembles by {args.attr.upper()} head_angle attribution "
      f"(session index {sess_idx}):")
for e in top_ensemble_idx:
    print(f"  Ensemble {e:2d}:  {args.attr} = {ha_attr[e]:.4f}")

# ── Filter to session ─────────────────────────────────────────────────────────
sess_mask = beh.index.map(lambda t: t[0]) == args.session
beh_sess  = beh[sess_mask]
spk_sess  = spk[sess_mask]

common   = beh_sess.index.intersection(spk_sess.index)
beh_sess = beh_sess.loc[common]
spk_sess = spk_sess.loc[common]

ens_act_sess = spk_sess.astype(float).values @ ensembles   # for z-score baseline

# ── Filter to trial ───────────────────────────────────────────────────────────
trial_mask = beh_sess["trial_id"] == args.trial_id
beh_trial  = beh_sess[trial_mask].copy()
spk_trial  = spk_sess[trial_mask].copy()

if beh_trial.empty:
    sys.exit(f"Trial {args.trial_id} not found in session '{args.session}'.")

print(f"\nTrial {args.trial_id}: {len(beh_trial)} frames")

sort_order = np.argsort(beh_trial["frame_pc_timestamp"].values)
beh_trial  = beh_trial.iloc[sort_order].copy()
spk_trial  = spk_trial.iloc[sort_order].copy()

t0        = beh_trial["frame_pc_timestamp"].values[0]
rel_times = (beh_trial["frame_pc_timestamp"].values - t0) / 1e6
T_total   = rel_times[-1]

# Compute ensemble activations for this trial and z-score for display
ens_trial = spk_trial.astype(float).values @ ensembles   # (n_frames, n_ensembles)
ens_trial_z = np.zeros_like(ens_trial)
for i in range(n_ensembles):
    mu, sd = ens_act_sess[:, i].mean(), ens_act_sess[:, i].std() + 1e-8
    ens_trial_z[:, i] = (ens_trial[:, i] - mu) / sd

ha_trial = beh_trial["head_angle"].astype(float).values

# ── Build track list: head_angle first, then top_k ensembles ─────────────────
track_labels  = ["head_angle"] + [f"Ensemble {e}" for e in top_ensemble_idx]
track_arrays  = [ha_trial]     + [ens_trial_z[:, e] for e in top_ensemble_idx]
track_attrs   = [None]         + [ha_attr[e] for e in top_ensemble_idx]
n_tracks      = len(track_labels)

# ── Video alignment (supervisor's method) ─────────────────────────────────────
# Sort the entire session by frame_pc_timestamp — the resulting row order
# matches the video frame order (same camera).  Each frame's video index is
# its position in this sorted sequence, found via searchsorted on the timestamp.
session_ts_sorted = np.sort(beh_sess["frame_pc_timestamp"].values)

def ts_to_video_frame(frame_pc_ts):
    """Map a behavioral frame's PC timestamp to its video frame index."""
    return int(np.searchsorted(session_ts_sorted, frame_pc_ts))

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Warning: could not open video at '{args.video}'. Video panel will be blank.")
    cap = None
    video_fps = 30.0
else:
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_fps:.2f} fps | {n_video_frames} frames")

def read_video_frame(data_idx):
    """Look up video frame via session-level timestamp order."""
    if cap is None:
        return None
    ts     = beh_trial["frame_pc_timestamp"].values[data_idx]
    target = min(ts_to_video_frame(ts), n_video_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ── Figure layout ─────────────────────────────────────────────────────────────
DARK_BG    = "#0f0f1a"
PANEL_BG   = "#13131f"
TEXT_COLOR = "#ccccdd"
SLIDER_COL = "#3355aa"
CURSOR_COL = "#ff4455"

track_h = 0.7
video_h = max(3.5, track_h * n_tracks)
fig_h   = video_h + 1.0
fig_w   = 16

fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)

row_heights = [1] * n_tracks + [0.18, 0.12]
gs = gridspec.GridSpec(
    n_tracks + 2, 2,
    figure=fig,
    width_ratios=[1.05, 1.5],
    height_ratios=row_heights,
    hspace=0.04, wspace=0.06,
    left=0.02, right=0.98,
    top=0.96, bottom=0.06,
)

ax_vid = fig.add_subplot(gs[:n_tracks, 0])
ax_vid.set_facecolor("black")
ax_vid.set_xticks([]); ax_vid.set_yticks([])
for sp in ax_vid.spines.values():
    sp.set_edgecolor("#2a2a4a")
ax_vid.set_title(f"Trial {args.trial_id}  |  {args.session}",
                 color=TEXT_COLOR, fontsize=9, pad=4)

first_frame = read_video_frame(0)
if first_frame is not None:
    vid_img = ax_vid.imshow(first_frame, aspect="auto")
else:
    vid_img = ax_vid.imshow(np.zeros((480, 640, 3), dtype=np.uint8), aspect="auto")

time_text = ax_vid.text(
    0.02, 0.97, "t = 0.00 s",
    transform=ax_vid.transAxes, color=TEXT_COLOR,
    fontsize=8, va="top", ha="left",
    bbox=dict(facecolor="#00000088", edgecolor="none", pad=2))

# head_angle gets a warm teal; ensembles get plasma
ha_color  = "#4ecdc4"
ens_colors = plt.cm.plasma(np.linspace(0.3, 0.9, args.top_k))

track_colors = [ha_color] + list(ens_colors)

# ── Trace axes ────────────────────────────────────────────────────────────────
trace_axes = []
WINDOW = args.window

for i, (label, arr, color, attr_val) in enumerate(
        zip(track_labels, track_arrays, track_colors, track_attrs)):
    ax = fig.add_subplot(gs[i, 1])
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")

    ax.plot(rel_times, arr, color=color, linewidth=0.9, alpha=0.9)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    mid = (lo + hi) / 2
    ax.set_yticks([lo, mid, hi])
    ax.set_yticklabels([f"{lo:.2g}", f"{mid:.2g}", f"{hi:.2g}"],
                       fontsize=5, color=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=5, length=2, pad=1)
    ax.axhline(0, color="#ffffff18", linewidth=0.5, zorder=0)
    ax.set_xlim(0, min(WINDOW, T_total))

    pretty = "Head Angle" if label == "head_angle" else \
             f"E{label.split()[-1]}  {args.attr}={attr_val:.3f}"
    ax.set_ylabel(pretty, color=color, fontsize=6.5, rotation=0,
                  labelpad=4, va="center", ha="right")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", which="both", left=False, right=True,
                   labelright=True, labelleft=False)

    if i < n_tracks - 1:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=7)
        ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=6, length=3)

    xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)
    cursor, = ax.plot([0, 0], [0, 1], color=CURSOR_COL, linewidth=1.2,
                      alpha=0.85, zorder=5, transform=xdata_yaxes, clip_on=False)
    trace_axes.append((ax, cursor))

# ── Slider + Buttons ──────────────────────────────────────────────────────────
ax_sl = fig.add_subplot(gs[n_tracks, :])
ax_sl.set_facecolor(DARK_BG); ax_sl.set_xticks([]); ax_sl.set_yticks([])
slider = Slider(ax_sl, "  t ", 0.0, T_total, valinit=0.0,
                color=SLIDER_COL, track_color="#222233")
slider.label.set_color(TEXT_COLOR)
slider.valtext.set_color(TEXT_COLOR)

btn_play_ax  = fig.add_axes([0.38, 0.01, 0.09, 0.035])
btn_speed_ax = fig.add_axes([0.50, 0.01, 0.09, 0.035])
btn_reset_ax = fig.add_axes([0.62, 0.01, 0.09, 0.035])
btn_play  = Button(btn_play_ax,  "▶  Play",  color="#1e3a6e", hovercolor="#2a4e99")
btn_speed = Button(btn_speed_ax, f"×{args.speed:.2f}", color="#1e4a2e", hovercolor="#2a6a3e")
btn_reset = Button(btn_reset_ax, "⟳  Reset", color="#3a1e1e", hovercolor="#5a2e2e")
for btn in (btn_play, btn_speed, btn_reset):
    btn.label.set_color("white"); btn.label.set_fontsize(8)

SPEED_CYCLE = [0.25, 0.5, 1.0, 2.0, 4.0]
state = {"playing": False, "t": 0.0, "speed": args.speed,
         "speed_idx": SPEED_CYCLE.index(1.0)}

def _update_display(t):
    t = float(np.clip(t, 0.0, T_total))
    state["t"] = t

    data_idx = min(int(np.searchsorted(rel_times, t, side="left")), len(rel_times) - 1)
    frame = read_video_frame(data_idx)
    if frame is not None:
        vid_img.set_data(frame)

    time_text.set_text(f"t = {t:.2f}s")

    half   = WINDOW / 2
    xleft  = max(0.0, t - half)
    xright = xleft + WINDOW
    if xright > T_total:
        xright = T_total; xleft = max(0.0, xright - WINDOW)

    for ax, cursor in trace_axes:
        cursor.set_xdata([t, t])
        ax.set_xlim(xleft, xright)

    slider.eventson = False
    slider.set_val(t)
    slider.eventson = True
    fig.canvas.draw_idle()

def on_slider(val):
    state["playing"] = False
    btn_play.label.set_text("▶  Play")
    _update_display(val)

def on_play(event):
    state["playing"] = not state["playing"]
    btn_play.label.set_text("⏸  Pause" if state["playing"] else "▶  Play")

def on_speed(event):
    state["speed_idx"] = (state["speed_idx"] + 1) % len(SPEED_CYCLE)
    state["speed"]     = SPEED_CYCLE[state["speed_idx"]]
    btn_speed.label.set_text(f"×{state['speed']:.2f}")

def on_reset(event):
    state["playing"] = False
    btn_play.label.set_text("▶  Play")
    _update_display(0.0)

slider.on_changed(on_slider)
btn_play.on_clicked(on_play)
btn_speed.on_clicked(on_speed)
btn_reset.on_clicked(on_reset)
fig.canvas.mpl_connect("key_press_event", lambda e: on_play(None) if e.key == " " else None)

INTERVAL_MS = max(20, int(1000 / video_fps))

def _step(frame_num):
    if not state["playing"]:
        return
    new_t = state["t"] + (1.0 / video_fps) * state["speed"]
    if new_t >= T_total:
        state["playing"] = False
        btn_play.label.set_text("▶  Play")
        new_t = T_total
    _update_display(new_t)

_update_display(0.0)

if args.save:
    out_path = args.save_path or f"head_angle_ensembles_trial{args.trial_id}.mp4"
    out_fps  = args.save_fps  or video_fps
    n_frames = len(rel_times)
    print(f"\nRendering {n_frames} frames → {out_path}  ({out_fps:.1f} fps, {args.save_dpi} dpi) …")
    writer = FFMpegWriter(fps=out_fps, bitrate=3000,
                          metadata=dict(title=f"Trial {args.trial_id}"))
    with writer.saving(fig, out_path, dpi=args.save_dpi):
        for i, t in enumerate(rel_times):
            _update_display(t)
            writer.grab_frame()
            if i % 60 == 0 or i == n_frames - 1:
                print(f"  {i:>5}/{n_frames}  ({100*i/max(n_frames-1,1):.0f}%)", end="\r", flush=True)
    print(f"\nSaved → {out_path}")
else:
    anim = FuncAnimation(fig, _step, interval=INTERVAL_MS, cache_frame_data=False)
    print(f"\nReady. Trial duration: {T_total:.2f}s | {n_tracks} tracks "
          f"(head_angle + {args.top_k} ensembles)")
    print("Press Space or click Play to start.\n")
    plt.show()

if cap is not None:
    cap.release()
