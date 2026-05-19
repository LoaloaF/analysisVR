#!/usr/bin/env python3
"""
watch_trial.py — Play a bodycam trial alongside behavioral features and neural activity.

Usage
-----
    # Default: trial 170, session 2025-01-26_21-48, no neurons
    python watch_trial.py

    # Show neurons by 0-based index (Unit0001 = 0, Unit0002 = 1, ...)
    python watch_trial.py --neurons 0 5 12

    # List available neuron column names and exit
    python watch_trial.py --list_neurons

    # Change trial or session
    python watch_trial.py --trial_id 170 --session 2025-01-26_21-48

    # Adjust video sync if needed (seconds in video where trial starts)
    python watch_trial.py --trial_start 3331

Controls
--------
    Play / Pause  — button or spacebar
    ×N button     — cycles playback speed (0.25×, 0.5×, 1×, 2×, 4×)
    Slider        — scrub to any point in the trial
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
parser.add_argument("--trial_id",    type=int,   default=170)
parser.add_argument("--session",     type=str,   default="2025-01-26_21-48")
parser.add_argument("--trial_start", type=float, default=55 * 60 + 30,
                    help="Trial start time in video in seconds (default: 55:31 = 3331)")
parser.add_argument("--video",       type=str,
                    default="./bodycam.mp4")
parser.add_argument("--base",        type=str,
                    default="./outputs/glm_input_data",
                    help="Directory containing the .npy data files")
parser.add_argument("--neurons",     type=int,   nargs="*", default=[],
                    help="0-based column indices of neurons to show (e.g. --neurons 0 5 12)")
parser.add_argument("--speed",       type=float, default=1.0,
                    help="Initial playback speed (default 1.0)")
parser.add_argument("--window",      type=float, default=6.0,
                    help="Visible time window in seconds (default 6)")
parser.add_argument("--list_neurons", action="store_true",
                    help="Print neuron column names with their indices and exit")
parser.add_argument("--save",      action="store_true",
                    help="Render to MP4 instead of opening the interactive window")
parser.add_argument("--save_path", type=str, default=None,
                    help="Output path (default: trial_<id>.mp4)")
parser.add_argument("--save_fps",  type=float, default=None,
                    help="Output FPS (default: matches video FPS)")
parser.add_argument("--save_dpi",  type=int,   default=120,
                    help="Output DPI (default: 120)")
args = parser.parse_args()

BEHAVIOR_FEATURES = [
    "frame_raw_500msMedian",
    "frame_YawPitch_abs_vel_sum_500msMedian",
    "frame_raw_abs_acc_500msMedian",
    "frame_YawPitch_abs_acc_sum_500msMedian",
    "forward_vs_rotation_corr",
    "head_angle",
    "head_angle_vel",
    "movement_energy_smooth5",
    "lick_detected",
    "track_zone_int",
    "cue_visible",
    "reward-sound_detected",
    "reward-valve-open_detected",
]

PRETTY_NAMES = {
    "frame_raw_500msMedian":                  "Forward Speed (500 ms median)",
    "frame_YawPitch_abs_vel_sum_500msMedian": "Yaw+Pitch Abs. Velocity (500 ms)",
    "frame_raw_abs_acc_500msMedian":          "Forward Abs. Acceleration (500 ms)",
    "frame_YawPitch_abs_acc_sum_500msMedian": "Yaw+Pitch Abs. Acceleration (500 ms)",
    "forward_vs_rotation_corr":               "Forward vs. Rotation Correlation",
    "head_angle":                             "Head Angle",
    "head_angle_vel":                         "Head Angular Velocity",
    "movement_energy_smooth5":                "Movement Energy (smoothed)",
    "lick_detected":                          "Lick Detected",
    "track_zone_int":                         "Track Zone",
    "cue_visible":                            "Cue Visible",
    "reward-sound_detected":                  "Reward Sound",
    "reward-valve-open_detected":             "Reward Valve Open",
}

# Binary features get a step-plot style
BINARY_FEATURES = {"lick_detected", "cue_visible", "reward-sound_detected", "reward-valve-open_detected"}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
base = args.base

beh_vals = np.load(f"{base}/behavior_glm_input.npy", allow_pickle=True)
beh_idx  = np.load(f"{base}/behavior_glm_input_index.npy", allow_pickle=True)
beh_cols = np.load(f"{base}/behavior_glm_input_columns.npy", allow_pickle=True)
beh = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)

spk_vals = np.load(f"{base}/fr_full.npy")
spk_idx  = np.load(f"{base}/fr_full_index.npy", allow_pickle=True)
spk_cols = np.load(f"{base}/fr_full_columns.npy", allow_pickle=True)
spk = pd.DataFrame(spk_vals, index=pd.Index(spk_idx), columns=spk_cols)

if args.list_neurons:
    print(f"\nAvailable neurons ({len(spk_cols)} total):")
    for i, name in enumerate(spk_cols):
        print(f"  {i:4d}  {name}")
    sys.exit(0)

spk.index = pd.MultiIndex.from_tuples(
    spk.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)

# Filter to session
sess_mask = beh.index.map(lambda t: t[0]) == args.session
beh = beh[sess_mask].copy()
spk = spk[sess_mask].copy()

if beh.empty:
    sys.exit(f"Session '{args.session}' not found in data.")

# Filter to trial
trial_mask = beh["trial_id"] == args.trial_id
# import pdb; pdb.set_trace()
beh_trial  = beh[trial_mask].copy()
spk_trial  = spk[trial_mask].copy()

if beh_trial.empty:
    sys.exit(f"Trial {args.trial_id} not found in session '{args.session}'.")

print(f"Trial {args.trial_id}: {len(beh_trial)} frames")

# Sort by timestamp to handle any dropped-frame reordering
sort_order = np.argsort(beh_trial["frame_pc_timestamp"].values)
beh_trial  = beh_trial.iloc[sort_order].copy()
spk_trial  = spk_trial.iloc[sort_order].copy()

t0        = beh_trial["frame_pc_timestamp"].values[0]
rel_times = (beh_trial["frame_pc_timestamp"].values - t0) / 1e6   # seconds
T_total   = rel_times[-1]

# Validate neuron indices
valid_neurons = []
for n in args.neurons:
    if 0 <= n < len(spk_cols):
        valid_neurons.append(n)
    else:
        print(f"  Warning: neuron index {n} out of range (0–{len(spk_cols)-1}), skipping.")

# Build arrays to plot
feat_names   = [f for f in BEHAVIOR_FEATURES if f in beh_trial.columns]
feat_arrays  = [beh_trial[f].astype(float).values for f in feat_names]
neuron_names = [str(spk_cols[n]) for n in valid_neurons]
neuron_arrays = [spk_trial.iloc[:, n].astype(float).values for n in valid_neurons]

all_names  = feat_names + neuron_names
all_arrays = feat_arrays + neuron_arrays
n_tracks   = len(all_names)

# ── Video ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Warning: could not open video at '{args.video}'. "
          "Video display will be blank.")
    cap = None
    video_fps = 30.0
else:
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {video_fps:.2f} | trial starts at {args.trial_start:.1f}s "
          f"(frame {int(args.trial_start * video_fps)})")

trial_start_vframe = int(args.trial_start * video_fps)

def read_video_frame(data_idx):
    """Map behavioral data row index → video frame number.

    Using the data index directly (rather than rel_t * video_fps) avoids
    accumulated drift between the PC-timestamp clock and the video clock.
    One behavioral row == one video frame because they are the same camera.
    """
    if cap is None:
        return None
    target = trial_start_vframe + int(data_idx)
    target = max(0, min(target, n_video_frames - 1))
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

# Track height in figure units
track_h    = 0.6   # inches per track
video_h    = max(3.5, track_h * n_tracks)
ctrl_h     = 0.9
fig_h      = video_h + ctrl_h + 0.6
fig_w      = 16

fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)

# Row heights: n_tracks equal rows + slider row + button row
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

# Video panel (left, spans all track rows)
ax_vid = fig.add_subplot(gs[:n_tracks, 0])
ax_vid.set_facecolor("black")
ax_vid.set_xticks([]); ax_vid.set_yticks([])
for sp in ax_vid.spines.values():
    sp.set_edgecolor("#2a2a4a")
ax_vid.set_title(
    f"Trial {args.trial_id}  |  {args.session}",
    color=TEXT_COLOR, fontsize=9, pad=4
)

first_frame = read_video_frame(0)
if first_frame is not None:
    vid_img = ax_vid.imshow(first_frame, aspect="auto")
else:
    vid_img = ax_vid.imshow(np.zeros((480, 640, 3), dtype=np.uint8), aspect="auto")

# Add a clock label in the video panel
time_text = ax_vid.text(
    0.02, 0.97, "t = 0.00 s",
    transform=ax_vid.transAxes, color=TEXT_COLOR,
    fontsize=8, va="top", ha="left",
    bbox=dict(facecolor="#00000088", edgecolor="none", pad=2)
)

# Color palettes
colors_feat  = plt.cm.Set2(np.linspace(0, 1, max(len(feat_names), 1)))
colors_neur  = plt.cm.plasma(np.linspace(0.3, 0.9, max(len(neuron_names), 1)))
all_colors   = (list(colors_feat[: len(feat_names)]) +
                list(colors_neur[: len(neuron_names)]))

# Trace axes (right column)
trace_axes = []   # list of (ax, cursor_line, ymin, ymax)
WINDOW = args.window

is_neuron_track = [False] * len(feat_names) + [True] * len(neuron_names)

for i, (label, arr, color, is_neuron) in enumerate(
        zip(all_names, all_arrays, all_colors, is_neuron_track)):
    ax = fig.add_subplot(gs[i, 1])
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")

    is_binary = label in BINARY_FEATURES
    if is_binary:
        ax.step(rel_times, arr, color=color, linewidth=0.9, alpha=0.85, where="post")
        ax.set_ylim(-0.15, 1.35)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["off", "on"], fontsize=5.5, color=TEXT_COLOR)
    else:
        ax.plot(rel_times, arr, color=color, linewidth=0.8, alpha=0.9)
        lo, hi = arr.min(), arr.max()
        mid    = (lo + hi) / 2
        ax.set_yticks([lo, mid, hi])
        ax.set_yticklabels(
            [f"{lo:.2g}", f"{mid:.2g}", f"{hi:.2g}"],
            fontsize=5, color=TEXT_COLOR
        )

    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=5, length=2, pad=1)
    ymin, ymax = ax.get_ylim()

    # Pretty label on left side
    pretty = (PRETTY_NAMES.get(label, label) if not is_neuron
              else f"FR  {label}")
    ax.set_ylabel(
        pretty, color=color,
        fontsize=6.5, rotation=0,
        labelpad=4, va="center", ha="right",
    )
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", which="both", left=False, right=True,
                   labelright=True, labelleft=False)

    # Faint horizontal zero-line for continuous traces
    if not is_binary:
        ax.axhline(0, color="#ffffff18", linewidth=0.5, zorder=0)

    ax.set_xlim(0, min(WINDOW, T_total))

    if i < n_tracks - 1:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=7)
        ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=6, length=3)

    # Blended transform: x in data coords, y in axes-fraction (0→1 = full height).
    # This means the cursor always spans the entire track regardless of ylim changes.
    xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)
    cursor, = ax.plot(
        [0, 0], [0, 1],
        color=CURSOR_COL, linewidth=1.2, alpha=0.85, zorder=5,
        transform=xdata_yaxes, clip_on=False,
    )
    trace_axes.append((ax, cursor))

# ── Slider ────────────────────────────────────────────────────────────────────
ax_sl = fig.add_subplot(gs[n_tracks, :])
ax_sl.set_facecolor(DARK_BG)
ax_sl.set_xticks([]); ax_sl.set_yticks([])

slider = Slider(ax_sl, "  t ", 0.0, T_total, valinit=0.0,
                color=SLIDER_COL, track_color="#222233")
slider.label.set_color(TEXT_COLOR)
slider.valtext.set_color(TEXT_COLOR)

# ── Buttons (using add_axes for precise placement) ────────────────────────────
btn_play_ax  = fig.add_axes([0.38, 0.01, 0.09, 0.035])
btn_speed_ax = fig.add_axes([0.50, 0.01, 0.09, 0.035])
btn_reset_ax = fig.add_axes([0.62, 0.01, 0.09, 0.035])

btn_play  = Button(btn_play_ax,  "▶  Play",  color="#1e3a6e", hovercolor="#2a4e99")
btn_speed = Button(btn_speed_ax, f"×{args.speed:.2f}", color="#1e4a2e", hovercolor="#2a6a3e")
btn_reset = Button(btn_reset_ax, "⟳  Reset", color="#3a1e1e", hovercolor="#5a2e2e")
for btn in (btn_play, btn_speed, btn_reset):
    btn.label.set_color("white")
    btn.label.set_fontsize(8)

# ── State ─────────────────────────────────────────────────────────────────────
SPEED_CYCLE = [0.25, 0.5, 1.0, 2.0, 4.0]
state = {
    "playing":   False,
    "t":         0.0,
    "speed":     args.speed,
    "speed_idx": SPEED_CYCLE.index(1.0),
}

def _update_display(t):
    t = float(np.clip(t, 0.0, T_total))
    state["t"] = t

    # Map time → nearest data row index, then to video frame
    data_idx = int(np.searchsorted(rel_times, t, side="left"))
    data_idx = min(data_idx, len(rel_times) - 1)
    frame = read_video_frame(data_idx)
    if frame is not None:
        vid_img.set_data(frame)

    # Clock
    mins  = int(args.trial_start + t) // 60
    secs  = (args.trial_start + t) % 60
    time_text.set_text(f"t = {t:.2f}s  ({mins}:{secs:05.2f})")

    # Half = WINDOW/2; keep cursor centred while possible
    half   = WINDOW / 2
    xleft  = max(0.0, t - half)
    xright = xleft + WINDOW
    if xright > T_total:
        xright = T_total
        xleft  = max(0.0, xright - WINDOW)

    for ax, cursor in trace_axes:
        cursor.set_xdata([t, t])
        ax.set_xlim(xleft, xright)

    # Suppress slider callback recursion
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

def on_key(event):
    if event.key == " ":
        on_play(None)

fig.canvas.mpl_connect("key_press_event", on_key)

# ── Animation loop ────────────────────────────────────────────────────────────
INTERVAL_MS = max(20, int(1000 / video_fps))   # target video frame rate

def _step(frame_num):
    if not state["playing"]:
        return
    dt    = (1.0 / video_fps) * state["speed"]
    new_t = state["t"] + dt
    if new_t >= T_total:
        state["playing"] = False
        btn_play.label.set_text("▶  Play")
        new_t = T_total
    _update_display(new_t)

_update_display(0.0)

if args.save:
    # ── Offline render ────────────────────────────────────────────────────────
    out_path  = args.save_path or f"trial_{args.trial_id}.mp4"
    out_fps   = args.save_fps  or video_fps
    n_frames  = len(rel_times)

    print(f"\nRendering {n_frames} frames → {out_path}  "
          f"({out_fps:.1f} fps, {args.save_dpi} dpi) …")

    writer = FFMpegWriter(fps=out_fps, bitrate=3000,
                          metadata=dict(title=f"Trial {args.trial_id}",
                                        artist="watch_trial.py"))
    with writer.saving(fig, out_path, dpi=args.save_dpi):
        for i, t in enumerate(rel_times):
            _update_display(t)
            writer.grab_frame()
            if i % 60 == 0 or i == n_frames - 1:
                pct = 100 * i / max(n_frames - 1, 1)
                print(f"  {i:>5}/{n_frames}  ({pct:.0f}%)", end="\r", flush=True)

    print(f"\nSaved → {out_path}")

else:
    # ── Interactive window ────────────────────────────────────────────────────
    anim = FuncAnimation(fig, _step, interval=INTERVAL_MS, cache_frame_data=False)
    print(f"\nReady. Trial duration: {T_total:.2f}s  |  {n_tracks} tracks "
          f"({len(feat_names)} behavior, {len(neuron_names)} neurons)")
    print("Press Space or click Play to start.\n")
    plt.show()

if cap is not None:
    cap.release()
