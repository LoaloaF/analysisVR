# import numpy as np
import json
import pandas as pd
import numpy as np
import os
import yaml
from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

import analytics_processing.modality_transformations as mT
import analytics_processing.analytics_utils as aU
from analysis_utils import device_paths
from analytics_processing.modality_loading import session_modality_from_nas
import mat73

def get_SesssionMetadata(session_fullfname):
    data = session_modality_from_nas(session_fullfname, "metadata")
    # drop nested json-like fields
    data = {k:v for k,v in data.items() 
            if k not in ["env_metadata","fsm_metadata","configuration","log_file_content"]}
    data = pd.Series(data, name=0).to_frame().T
    return data

def get_AnimalPose(session_fullfname):
    pass    
    
def get_FacecamPoses(session_fullfname):
    data = session_modality_from_nas(session_fullfname, "facecam_packages")
    nas_dir, _, _ = device_paths()
    project_path = os.path.join(nas_dir, "pose_estimation", "ratvt_butt-Haotian-2024-10-04") # modify based on the model
    project_yaml_path = os.path.join(project_path, "config.yaml")
    # modify the project_path in the yaml file based on the current nas_dir
    with open(project_yaml_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    project_base_path = yaml_content.get('project_base_path')
    modify_project_path = os.path.join(nas_dir, project_base_path)
    yaml_content['project_path'] = modify_project_path

    # Write the updated content back to the YAML file
    with open(project_yaml_path, 'w') as file:
        yaml.safe_dump(yaml_content, file)

    session_dir = os.path.dirname(session_fullfname)
    file_name = os.path.basename(session_fullfname)

    all_files = [file for file in os.listdir(session_dir)]
    # check if there is already DLC csv files
    dlc_csv_files = [file for file in all_files if "DLC" in file and file.endswith(".csv")]

    if not dlc_csv_files:
        # check if there is already facecam mp4 for analysis
        if "facecam.mp4" not in all_files:
            aU.hdf5_frames2mp4(session_dir, file_name)
        
        video_path = os.path.join(session_dir, "facecam.mp4")
        import deeplabcut # only done here to avoid unnecessary import
        deeplabcut.analyze_videos(
            config=f"{project_path}/config.yaml",
            videos=video_path,
            videotype='mp4',  # Replace with the video file format if different
            batchsize=8,
            save_as_csv=True,  # Save as intermediate CSVs for inspection
        )

        all_files = [file for file in os.listdir(session_dir)]
        dlc_csv_files = [file for file in all_files if "DLC" in file and file.endswith(".csv")]
    
    # read the csv file and reformat
    df_pose = pd.read_csv(os.path.join(session_dir, dlc_csv_files[0]), low_memory=False)
    df_pose.drop(columns=['scorer'], inplace=True)
    # reset the column names
    new_columns = df_pose.iloc[0] + '_' + df_pose.iloc[1]
    df_pose.columns = new_columns
    df_pose = df_pose.iloc[2:]
    df_pose.reset_index(drop=True, inplace=True)
    df_pose = df_pose.add_prefix("facecam_pose_")

    df_pose["facecam_image_pc_timestamp"] = data["facecam_image_pc_timestamp"]
    df_pose["facecam_image_ephys_timestamp"] = data["facecam_image_ephys_timestamp"]
    return df_pose 

def get_BehaviorEvents(session_fullfname):
    # deprecated, use get_Portenta
    return get_Portenta(session_fullfname)

def get_Portenta(session_fullfname):
    # event data
    cols = ("event_pc_timestamp","event_value","event_name",
            "event_ephys_timestamp","event_ephys_patched")
    eventdata = session_modality_from_nas(session_fullfname, "event", columns=cols)

    eventdata = mT.event_modality_stoplick2startlick(eventdata)
    # check the patched indicator
    if "event_ephys_patched" not in eventdata.columns:
        eventdata["event_ephys_patched"] = np.nan
    renamer = {col_name: col_name.replace("event", "portenta") 
               for col_name in eventdata.columns}
    eventdata.rename(columns=renamer, inplace=True)
    
    # ball velocity data    
    cols = ("ballvelocity_pc_timestamp","ballvelocity_ephys_timestamp",
            "ballvelocity_raw","ballvelocity_yaw","ballvelocity_pitch",
            "ballvelocity_ephys_patched")
    balldata = session_modality_from_nas(session_fullfname, "ballvelocity", columns=cols)
    # split single ball sensor reading event into 3 separate rows for RAW, YAW, PITCH
    raw_data, yaw_data, pitch_data = mT.ballvel_modality_split_ball_velocity(balldata)
    # for merging all tables at the end, start with eventdata
    portenta = [eventdata]
    for d in (yaw_data, raw_data, pitch_data):
        renamer = {col_name: col_name.replace("ballvelocity", "portenta") 
                for col_name in d.columns}
        portenta.append(d.rename(columns=renamer))
    
    # merge and return
    portenta = pd.concat(portenta, ignore_index=True)
    portenta = portenta.sort_values(by=['portenta_pc_timestamp', 'portenta_name'])
    return portenta.reset_index(drop=True)
    
def get_UnityFramewise(session_fullfname):
    framedata = session_modality_from_nas(session_fullfname, "unity_frame")
    metad = session_modality_from_nas(session_fullfname, "metadata")
    
    if metad['paradigm_id'] in (800, 1100):
        track_details = json.loads(metad['track_details'])
        
        # insert z position bin (1cm)
        from_z_position, to_z_position = mT.unity_modality_track_spatial_bins(framedata)
        framedata['from_z_position_bin'] = from_z_position
        framedata['to_z_position_bin'] = to_z_position

        # insert column with string which zone in at this frame                
        zone = mT.unity_modality_track_zones(framedata, track_details)
        framedata['zone'] = zone 
        
        # insert cue/ trial type, outcome etc, denormalize 
        unity_trial = session_modality_from_nas(session_fullfname, "unity_trial")
        trials_variable = session_modality_from_nas(session_fullfname, "paradigm_variable")
        framedata = pd.merge(framedata, trials_variable, on='trial_id', how='left')
        framedata = pd.merge(framedata, unity_trial, on='trial_id', how='left')
        
        # insert velocity and acceleration
        vel, acc = mT.unity_modality_track_kinematics(framedata)
        framedata = pd.concat([framedata, vel, acc], axis=1)
    return framedata

# TODO: should move from agg_modalities2analytic to integr_analytics, using unity_FrameWise
# right now, this reproduces what we had before, future should include kinematics
# calculations in UNityFrameWise, and could also not use metadata, and instead 
# do groupby on zone column in UnityFrameWise
def get_UnityTrialwiseMetrics(session_fullfname):
    # merge trial data with trial variables (cue, required staytime etc)
    trialdata = session_modality_from_nas(session_fullfname, "unity_trial")
    trials_variable = session_modality_from_nas(session_fullfname, "paradigm_variable")
    trialdata = pd.merge(trialdata, trials_variable, on='trial_id', how='left')
    
    # for track paradigms, calculate staytimes and other kinematic metrics 
    # in relevant zones using unity frames
    metad = session_modality_from_nas(session_fullfname, "metadata")
    if metad['paradigm_id'] in (800, 1100):
        track_details = json.loads(metad['track_details'])
        cols = ('trial_id', "frame_z_position", 'frame_pc_timestamp')
        unity_frames = session_modality_from_nas(session_fullfname, "unity_frame",
                                                columns=cols)
        staytimes = mT.calc_staytimes(trialdata, unity_frames, track_details)
        trialdata = pd.concat([trialdata, staytimes], axis=1)
    
    return trialdata


def get_Spikes(session_fullfname):
    session_dir = os.path.dirname(session_fullfname)
    analytics_dir = os.path.join(session_dir, "session_analytics")
    
    ephys_res = [file for file in os.listdir(analytics_dir) if "ephys" in file and file.endswith("_res.mat")]
    
    if not ephys_res:
        Logger().logger.warning(f"No ephys file found in {analytics_dir}")
        return None
    
    ephys_res = os.path.join(analytics_dir, ephys_res[0])
    ephys_res_mat = mat73.loadmat(ephys_res)

    # TODO: include more fields to this parquet
    clusters = ephys_res_mat["spikeClusters"] 
    spikeTimes = ephys_res_mat["spikeTimes"] * 50 # TODO: check if this is correct to multiply 50
    cluster_sites = ephys_res_mat["clusterSites"]

    spikes = pd.DataFrame({
    "cluster_id": clusters,
    "spike_time": spikeTimes
    })
    
    # Create a mapping from cluster to site
    cluster_to_site = {cluster_id: site_id for cluster_id, site_id in enumerate(cluster_sites)}
    spikes['site_id'] = spikes['cluster_id'].map(cluster_to_site)

    return spikes


