import os
import subprocess
import threading

import pandas as pd
import numpy as np

import cv2
import h5py

from CustomLogger import CustomLogger as Logger

def hdf5_frames2mp4_gpu(merged_fullfname, gpu_id=0, camera_names=None):
    """Render video from HDF5 frames using NVIDIA GPU acceleration via FFmpeg NVENC."""
    
    def _calc_fps(packages, cam_name):
        timestamps = packages[f'{cam_name}_image_pc_timestamp']
        diffs = np.diff(timestamps).astype(float)
        # round diffs to nearest integer (microsecond precision) so we can compute a mode robustly
        diffs_rounded = np.round(diffs).astype(np.int64)
        vals, counts = np.unique(diffs_rounded, return_counts=True)
        mode_us = vals[np.argmax(counts)]
        return np.round(1 / (mode_us / 1e6), 0)

    def _create_ffmpeg_writer(out_fullfname, fps, width, height, is_color=True):
        """Create FFmpeg subprocess with NVENC hardware encoding."""
        pix_fmt = 'bgr24' if is_color else 'gray'
        
        cmd = [
            'ffmpeg',
            '-y',  # overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', pix_fmt,
            '-r', str(fps),
            '-i', '-',  # read from pipe
            '-an',  # no audio
            # NVENC hardware encoding options
            '-c:v', 'h264_nvenc',
            '-gpu', str(gpu_id),
            '-preset', 'p4',  # balanced quality/speed (p1=fastest, p7=highest quality)
            '-tune', 'hq',  # high quality tuning
            '-rc', 'vbr',  # variable bitrate
            '-cq', '23',  # constant quality level (lower = better, 0-51)
            '-b:v', '0',  # let CQ control bitrate
            '-pix_fmt', 'yuv420p',  # output pixel format for compatibility
            out_fullfname
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        # Drain stderr in background thread to prevent pipe buffer from filling and blocking
        def drain_stderr(pipe):
            try:
                for line in pipe:
                    pass  # discard output
            except:
                pass
        
        threading.Thread(target=drain_stderr, args=(proc.stderr,), daemon=True).start()
        return proc

    def render_video(cam_name):
        ffmpeg_proc = None  # Initialize before try block to fix UnboundLocalError
        
        with h5py.File(merged_fullfname, 'r') as merged_file:
            try:
                packages = pd.read_hdf(merged_fullfname, key=f'{cam_name}_packages')
                fps = _calc_fps(packages, cam_name)

                frame_keys = list(merged_file[f"{cam_name}_frames"].keys())
                n_frames = len(frame_keys)
                L.logger.info(f"Rendering {cam_name} (n={n_frames:,} at {fps} FPS) using GPU...")
                
                for i, (frame_key, pack) in enumerate(zip(frame_keys, packages.iterrows())):
                    frame = merged_file[f"{cam_name}_frames"][frame_key][()]
                    frame = cv2.imdecode(np.frombuffer(frame.tobytes(), np.uint8), 
                                         cv2.IMREAD_COLOR)
                    
                    pack_id = pack[1][f"{cam_name}_image_id"]
                    
                    if i == 0:
                        height, width = frame.shape[:2]
                        is_color = len(frame.shape) == 3
                        out_fullfname = os.path.join(output_dir, f'{cam_name}.mp4')
                        ffmpeg_proc = _create_ffmpeg_writer(out_fullfname, fps, width, height, is_color)
                        prv_pack_id = pack_id - 1
                    
                    # insert black frames if package ID is discontinuous
                    gap = pack_id - prv_pack_id
                    if gap != 1:
                        n_missing = int(gap - 1)
                        L.logger.warning(f"Package ID discontinuous; gap was {gap}. "
                                         f"Inserting {n_missing} black frame(s).")
                        black_frame = np.zeros_like(frame).tobytes()
                        for _ in range(n_missing):
                            ffmpeg_proc.stdin.write(black_frame)
                    
                    print(f"Writing frame {i+1}/{n_frames} (package ID: {pack_id}, prv pack id: {prv_pack_id})... ", end='\r')
                    ffmpeg_proc.stdin.write(frame.tobytes())
                    prv_pack_id = pack_id
                    
                    # log progress
                    if n_frames >= 10 and i % (n_frames // 10) == 0:
                        print(f"0{i / n_frames * 100:.0f}% done...", end='\r')
                
                # cleanup
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait()
                    
                    if ffmpeg_proc.returncode != 0:
                        stderr = ffmpeg_proc.stderr.read().decode()
                        L.logger.error(f"FFmpeg error: {stderr}")
                    else:
                        L.logger.info(f"Successfully rendered {cam_name} video!")
                        
            except Exception as e:
                L.logger.error(f"Failed to render {cam_name} video: {e}")
                # print hdf5 file keys
                L.logger.error(f"HDF5 keys: {list(merged_file.keys())}")
                
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.kill()
                return
        return True

    L = Logger()
    L.logger.info(f"Rendering videos from HDF5 files in {os.path.dirname(merged_fullfname)} (GPU accelerated)")
    output_dir = os.path.join(os.path.dirname(merged_fullfname), 'rendered_videos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if camera_names is None:
        camera_names = ["facecam", "bodycam", "unitycam", "ttlcam2", "ttlcam3", "ttlcam4"]
        
    results = {}
    for cam in camera_names:
        # check if mp4 altready exists
        out_fullfname = os.path.join(output_dir, f'{cam}.mp4')
        if os.path.exists(out_fullfname):
            L.logger.info(f"Video for {cam} already exists, check if it's valid...")
            #open mp4 file to check if it's valid
            try:                
                cap = cv2.VideoCapture(out_fullfname)
                if not cap.isOpened():
                    L.logger.warning(f"Existing video for {cam} is not valid, will attempt to re-render.")
                else:
                    L.logger.info(f"Existing video for {cam} is valid, skipping rendering.")
                    continue
            except Exception as e:
                L.logger.warning(f"Could not open existing video for {cam} due to error: {e}. Will attempt to re-render.")
            
        L.logger.info(f"Rendering {cam}...")
        result = render_video(cam)
        results[cam] = result
    return results



# def jpglist_to_mp4_gpu(jpg_list, output_path, fps=30, gpu_id=0):
#     """
#     Render video from a list of JPG file paths using NVIDIA GPU acceleration.
    
#     Args:
#         jpg_list: List of paths to JPG files (in order)
#         output_path: Output MP4 file path
#         fps: Frames per second for output video
#         gpu_id: NVIDIA GPU device ID to use
#     """
#     L = Logger()
    
#     if not jpg_list:
#         L.logger.error("Empty JPG list provided")
#         return False
    
#     # Read first frame to get dimensions
#     first_frame = cv2.imread(jpg_list[0])
#     if first_frame is None:
#         L.logger.error(f"Could not read first frame: {jpg_list[0]}")
#         return False
    
#     height, width = first_frame.shape[:2]
#     is_color = len(first_frame.shape) == 3
#     pix_fmt = 'bgr24' if is_color else 'gray'
    
#     L.logger.info(f"Rendering {len(jpg_list):,} frames at {fps} FPS ({width}x{height}) using GPU...")
    
#     cmd = [
#         'ffmpeg',
#         '-y',
#         '-f', 'rawvideo',
#         '-vcodec', 'rawvideo',
#         '-s', f'{width}x{height}',
#         '-pix_fmt', pix_fmt,
#         '-r', str(fps),
#         '-i', '-',
#         '-an',
#         '-c:v', 'h264_nvenc',
#         '-gpu', str(gpu_id),
#         '-preset', 'p4',
#         '-tune', 'hq',
#         '-rc', 'vbr',
#         '-cq', '23',
#         '-b:v', '0',
#         '-pix_fmt', 'yuv420p',
#         output_path
#     ]
    
#     ffmpeg_proc = subprocess.Popen(
#         cmd,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.PIPE
#     )
    
#     n_frames = len(jpg_list)
#     try:
#         for i, jpg_path in enumerate(jpg_list):
#             frame = cv2.imread(jpg_path)
#             if frame is None:
#                 L.logger.warning(f"Could not read frame {i}: {jpg_path}, inserting black frame")
#                 frame = np.zeros_like(first_frame)
            
#             ffmpeg_proc.stdin.write(frame.tobytes())
            
#             if n_frames >= 10 and i % (n_frames // 10) == 0:
#                 print(f"{i / n_frames * 100:.0f}% done...", end='\r')
        
#         ffmpeg_proc.stdin.close()
#         ffmpeg_proc.wait()
        
#         if ffmpeg_proc.returncode != 0:
#             stderr = ffmpeg_proc.stderr.read().decode()
#             L.logger.error(f"FFmpeg error: {stderr}")
#             return False
        
#         L.logger.info(f"Successfully rendered video to {output_path}")
#         return True
        
#     except Exception as e:
#         L.logger.error(f"Failed to render video: {e}")
#         ffmpeg_proc.stdin.close()
#         ffmpeg_proc.kill()
#         return False
    
def link_dlc_training_data(session_fullfname, link_to_dir, cam_name='facecam'):
    L = Logger()
    if link_to_dir is None:
        L.logger.error("No --link_to_dir provided for DLC training prep")
        return
    
    # check if rendered_videos subdir exists
    rendered_videos_dir = os.path.join(os.path.dirname(session_fullfname), 'rendered_videos')
    if not os.path.exists(rendered_videos_dir):
        L.logger.error(f"No rendered_videos directory found for session: {rendered_videos_dir}")
        return
    # check if cam video exists
    cam_video_path = os.path.join(rendered_videos_dir, f'{cam_name}.mp4')
    if not os.path.exists(cam_video_path):
        L.logger.error(f"No {cam_name} video found for session: {cam_video_path}")
        return
    
    # session name
    session_name = os.path.basename(session_fullfname)[:-5]  # remove .hdf5
    linked_video_path = os.path.join(link_to_dir, f'{session_name}_{cam_name}.mp4')

    if not os.path.exists(link_to_dir):
        os.makedirs(link_to_dir)
    if not os.path.exists(linked_video_path):
        os.link(cam_video_path, linked_video_path)
        L.logger.info(f"Created link for DLC training data: {os.path.basename(linked_video_path)}")
    else:
        L.logger.info(f"Link already exists: {os.path.basename(linked_video_path)}")

    