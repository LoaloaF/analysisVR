import numpy as np
# import cv2
import h5py
import os
import pandas as pd


def hdf5_frames2mp4(session_dir, merged_fname):    
    def _calc_fps(packages, cam_name):
        timestamps = packages[f'{cam_name}_image_pc_timestamp']
        return np.round(1 / np.mean(np.diff(timestamps)) * 1e6, 0)

    def _create_video_writer(cam_name, fps, frame_shape):
        out_fullfname = os.path.join(session_dir, f'{cam_name}.mp4')
        out_dims = frame_shape[1], frame_shape[0] # flip dims for cv2
        isColor = True if len(frame_shape) == 3 else False
        return cv2.VideoWriter(out_fullfname, cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, out_dims, isColor=isColor)
    
    def render_video(cam_name):
        merged_fullfname = os.path.join(session_dir, merged_fname)
        with h5py.File(merged_fullfname, 'r') as merged_file:
            try:
                packages = pd.read_hdf(merged_fullfname, key=f'{cam_name}_packages')
                fps = _calc_fps(packages, cam_name)

                frame_keys = merged_file[f"{cam_name}_frames"].keys()
                # L.logger.info(f"Rendering {cam_name} (n={len(frame_keys):,})...")
                print(f"Rendering {cam_name} (n={len(frame_keys):,})...")
                for i, (frame_key, pack) in enumerate(zip(frame_keys, packages.iterrows())):
                    frame = merged_file[f"{cam_name}_frames"][frame_key][()]
                    frame = cv2.imdecode(np.frombuffer(frame.tobytes(), np.uint8), 
                                         cv2.IMREAD_COLOR) 
                    
                    pack_id = pack[1][f"{cam_name}_image_id"]
                    if i == 0:
                        writer = _create_video_writer(cam_name, fps, frame.shape)
                        prv_pack_id = pack_id-1
                    
                    # insert black frame if package ID is discontinuous
                    if pack_id != prv_pack_id+1:
                        # L.logger.warning(f"Package ID discontinuous; gap was "
                        #                  f"{pack_id - prv_pack_id}.  Inserting"
                        #                  f" black frame.")
                        writer.write(np.zeros_like(frame))
                    else:
                        writer.write(frame)
                    prv_pack_id = pack_id
                    
                    # log progress
                    # if i % (len(frame_keys)//10) == 0:
                    #     print(f"{i/len(frame_keys)*100:.0f}% done...", end='\r')
                # L.logger.info(f"Sucessfully rendered {cam_name} video!")
                print(f"Sucessfully rendered {cam_name} video!")
            # keys in hdf5 file may very well not exist
            except Exception as e:
                # L.logger.error(f"Failed to render {cam_name} video: {e}")
                print("Failed to render video")
                return
    render_video("facecam")