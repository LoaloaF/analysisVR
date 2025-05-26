# import numpy as np
import json
import pandas as pd
import numpy as np
import os
# import yaml
from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

import analytics_processing.modality_transformations as mT
import analytics_processing.analytics_utils as aU
from analytics_processing.analytics_constants import device_paths
import mat73

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.modality_loading import get_modality_summary
    
def get_SesssionMetadata(session_fullfname):
    data = session_modality_from_nas(session_fullfname, "metadata")
    data = pd.Series(data, name=0)
    
    # append a summary of existing modalities for this session
    modality_summary = get_modality_summary(session_fullfname)
    data = pd.concat([data, modality_summary], axis=0)
    return data.to_frame().T

def get_TrackKinematics(session_fullfname):
    framedata = session_modality_from_nas(session_fullfname, "unity_frame")
    metad = session_modality_from_nas(session_fullfname, "metadata")
    
    if metad['paradigm_id'] in (800, 1100):
        track_details = json.loads(metad['track_details'])
        
        # insert z position bin (1cm)
        from_z_position, to_z_position = mT.unity_modality_track_spatial_bins(framedata)
        framedata['from_cm_position_bin'] = from_z_position
        framedata['to_cm_position_bin'] = to_z_position

        # insert column with string which zone in at this frame                
        zone = mT.unity_modality_track_zones(framedata, track_details)
        framedata['track_zone'] = zone 
        
        # insert velocity and acceleration from unity frame data
        vel, acc = mT.unity_modality_track_kinematics(framedata)
        
        # insert ball velocity and yaw/pitch data from portenta
        cols = ("ballvelocity_package_id","ballvelocity_raw","ballvelocity_yaw",
                "ballvelocity_pitch",)
        balldata = session_modality_from_nas(session_fullfname, "ballvelocity", 
                                             columns=cols)
        raw_yaw_pitch = mT.frame_wise_ball_velocity(framedata, balldata)
        
        # merge all data        
        framedata = pd.concat([framedata, vel, acc, raw_yaw_pitch], axis=1)
        # track has only z changes, renmae the column
        framedata.rename(columns={"frame_z_position": "frame_position",
                                  "frame_z_velocity": "frame_velocity",
                                  "frame_z_acceleration": "frame_acceleration"}, 
                         inplace=True)
        # angle data is not needed for track, blinker is not used anywhere
        framedata.drop(columns=['frame_angle', 'frame_blinker', 'frame_x_position'], 
                       inplace=True)
    return framedata

def get_BehaviorTrialwise(session_fullfname, track_kinematics):
    # merge trial data with trial variables (cue, required staytime etc)
    trialdata = session_modality_from_nas(session_fullfname, "unity_trial")
    trials_variable = session_modality_from_nas(session_fullfname, "paradigm_variable")
    trialdata = pd.merge(trialdata, trials_variable, on='trial_id', how='left')
    # for track paradigms, calculate staytimes and other kinematic metrics 
    # in relevant zones using unity frames
    metad = session_modality_from_nas(session_fullfname, "metadata")
    if metad['paradigm_id'] in (800, 1100):
        metrics = mT.calc_trialwise_metrics(trialdata, track_kinematics, 
                                            trials_variable)
        behavior_trialwise = pd.concat([trialdata, metrics], axis=1)
    return behavior_trialwise

def get_BehaviorEvents(session_fullfname, trialwise_data):
    # read the event data from hdf5
    cols = ("event_pc_timestamp","event_value","event_name",
            "event_ephys_timestamp","event_ephys_patched")
    eventdata = session_modality_from_nas(session_fullfname, "event", columns=cols)

    # transform lick event to common format starttime + value (=duration)
    eventdata = mT.event_modality_stoplick2startlick(eventdata)
    
    # TODO: reward consumption: calculate from R, L and V events, 
    # add reward-available, reward-consumed events

    # check the patched indicator
    if "event_ephys_patched" not in eventdata.columns:
        eventdata["event_ephys_patched"] = np.nan

    event_fullnames = {'R': 'reward-valve-open', 'L': 'lick', 'S': 'reward-sound',
                       'V': 'reward-removed', 'P': 'airpuff', 'C': 'camera-frame'}
    eventdata['event_name_full'] = eventdata['event_name'].map(event_fullnames)
    
    # assign trials for merging with trialwise data
    trial_interval = pd.IntervalIndex.from_arrays(
        trialwise_data.pop('trial_start_pc_timestamp'),
        trialwise_data.pop('trial_end_pc_timestamp'), closed='both')
    # assign each event to one of the intervals, return -1 if not in any interval
    assignm = trial_interval.get_indexer(eventdata['event_pc_timestamp'])
    trial_assignm = trialwise_data.trial_id.iloc[assignm]
    trial_assignm[assignm == -1] = np.nan  # assign NaN for events in ITI
    eventdata['trial_id'] = trial_assignm.values
    
    # merge with trialwise data, cue, outcome, choice
    eventdata = pd.merge(eventdata, trialwise_data,
                         on='trial_id', how='left')
    return eventdata    


# TODO: implement this function
def get_BehaviorPose(session_fullfname, track_kinematics):
    pass    
# def get_FacecamPoses(session_fullfname):
#     data = session_modality_from_nas(session_fullfname, "facecam_packages")
#     nas_dir, _, _ = device_paths()
#     project_path = os.path.join(nas_dir, "pose_estimation", "ratvt_butt-Haotian-2024-10-04") # modify based on the model
#     project_yaml_path = os.path.join(project_path, "config.yaml")
#     # modify the project_path in the yaml file based on the current nas_dir
#     with open(project_yaml_path, 'r') as file:
#         yaml_content = yaml.safe_load(file)

#     project_base_path = yaml_content.get('project_base_path')
#     modify_project_path = os.path.join(nas_dir, project_base_path)
#     yaml_content['project_path'] = modify_project_path

#     # Write the updated content back to the YAML file
#     with open(project_yaml_path, 'w') as file:
#         yaml.safe_dump(yaml_content, file)

#     session_dir = os.path.dirname(session_fullfname)
#     file_name = os.path.basename(session_fullfname)

#     all_files = [file for file in os.listdir(session_dir)]
#     # check if there is already DLC csv files
#     dlc_csv_files = [file for file in all_files if "DLC" in file and file.endswith(".csv")]

#     if not dlc_csv_files:
#         # check if there is already facecam mp4 for analysis
#         if "facecam.mp4" not in all_files:
#             aU.hdf5_frames2mp4(session_dir, file_name)
        
#         video_path = os.path.join(session_dir, "facecam.mp4")
#         import deeplabcut # only done here to avoid unnecessary import
#         deeplabcut.analyze_videos(
#             config=f"{project_path}/config.yaml",
#             videos=video_path,
#             videotype='mp4',  # Replace with the video file format if different
#             batchsize=8,
#             save_as_csv=True,  # Save as intermediate CSVs for inspection
#         )

#         all_files = [file for file in os.listdir(session_dir)]
#         dlc_csv_files = [file for file in all_files if "DLC" in file and file.endswith(".csv")]
    
#     # read the csv file and reformat
#     df_pose = pd.read_csv(os.path.join(session_dir, dlc_csv_files[0]), low_memory=False)
#     df_pose.drop(columns=['scorer'], inplace=True)
#     # reset the column names
#     new_columns = df_pose.iloc[0] + '_' + df_pose.iloc[1]
#     df_pose.columns = new_columns
#     df_pose = df_pose.iloc[2:]
#     df_pose.reset_index(drop=True, inplace=True)
#     df_pose = df_pose.add_prefix("facecam_pose_")

#     df_pose["facecam_image_pc_timestamp"] = data["facecam_image_pc_timestamp"]
#     df_pose["facecam_image_ephys_timestamp"] = data["facecam_image_ephys_timestamp"]
#     return df_pose 