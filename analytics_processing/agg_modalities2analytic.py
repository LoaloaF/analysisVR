# import numpy as np
import json
import pandas as pd
import numpy as np
import os
from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

import analytics_processing.modality_transformations as mT
import analytics_processing.analytics_utils as aU
import analytics_processing.analytics_constants as aC

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.modality_loading import get_modality_summary
from general_processing.camera_helpers import hdf5_frames2mp4_gpu
from general_processing.pose_features import add_kinematic_features, kinematic_feature_summary

try:
    # may fail, DLC analytic computation of pose should use differnt conda env
    import yaml
    import deeplabcut
except ImportError as e:
    print(f"Warning: {e}. DeepLabCut-related functionality will not work. Please install deeplabcut and pyyaml to compute the BehaviorPose analytic.")

    
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
        # track_details = json.loads(metad['track_details'])
        # don't trust the zone definitions from metadata, use hardcoded ones
        track_details = {
            'startZone': (-169,-120),
            'visibleCue': (-120,-80),
            'nextToCue': (-80,10),
            'afterCue': (10, 50),
            'reward1Zone': (50,110),
            'bewteenRewardZones': (110,170),
            'reward2Zone': (170,230),
            'endZone': (230,260),
            'ITI': (260,265),
        }
        trackzone_int_ordered = {
            "startZone": 0,
            "visibleCue": 1,
            "nextToCue": 2,
            "afterCue": 3,
            "reward1Zone": 4,
            "bewteenRewardZones": 5,
            "reward2Zone": 6,
            "endZone": 7,
            "ITI": 8,
        }
        
        # paradigm start position is 0, no movement here, set to nan
        first_frame_non_0 = framedata.index[framedata['frame_z_position'] != 0][0]
        framedata.loc[:first_frame_non_0 - 1, 'frame_z_position'] = np.nan
        framedata.loc[:first_frame_non_0 - 1, 'trial_id'] = np.nan
        
        # add the fps column, shifted so that it indicates the length 
        framedata['fps'] = 1 / (framedata['frame_pc_timestamp'].diff()/ 1e6) # in seconds
        
        # insert column with string which zone in at this frame
        zone = mT.unity_modality_track_zones(framedata, track_details)
        framedata['track_zone'] = zone 
        framedata['track_zone_int'] = zone.map(lambda x: trackzone_int_ordered.get(x))
        
        # fix trial_id in ITI zones, should be same as ending trial
        trial_id = mT.fix_trial_id_in_ITI(framedata[['track_zone', 'trial_id', 'frame_id']])
        framedata['trial_id'] = trial_id
        
        # in ITI zone frame_z_position is meaningless, set to nan
        framedata.loc[(framedata['track_zone'] == 'ITI').values, 'frame_z_position'] = np.nan
        
        # insert z position bin (1cm)
        from_z_position, to_z_position = mT.unity_modality_track_spatial_bins(framedata)
        framedata['from_cm_position_bin'] = from_z_position
        framedata['to_cm_position_bin'] = to_z_position

        
        # insert velocity and acceleration from unity frame data
        vel, acc = mT.unity_modality_track_kinematics(framedata)
        
        # insert ball velocity and yaw/pitch data from portenta
        cols = ("ballvelocity_package_id","ballvelocity_raw","ballvelocity_yaw",
                "ballvelocity_pitch",)
        balldata = session_modality_from_nas(session_fullfname, "ballvelocity", 
                                             columns=cols)
        # sum over ryp pacakges that were used in a unity frame
        raw_yaw_pitch = mT.frame_wise_ball_velocity(framedata, balldata)
        
        # convert from cm/frame length to cm/s
        raw_yaw_pitch *= framedata.loc[raw_yaw_pitch.index,'fps'].values.reshape(-1,1)
        
        # for acceleration calculation
        tstamps = framedata.loc[raw_yaw_pitch.index,'frame_pc_timestamp'].values * 1e-6 # convert to seconds
        
        # caluclate additional kinematic features: smoothed velocity and acceleration 
        # for yaw and pitch, sum of absolute velocity and acceleration across yaw 
        # and pitch (off-rotation), sum of absolute velocity and 
        # acceleration across all 3 dimensions (for reward thresholding)
        more_kinem = mT.calculate_additional_kinematic_features(raw_yaw_pitch, tstamps)

        framedata = pd.concat([framedata, vel, acc, raw_yaw_pitch, more_kinem], axis=1)
                
        # track has only z changes, rename the column
        framedata.rename(columns={"frame_z_position": "frame_position",
                                  "frame_z_velocity": "frame_velocity",
                                  "frame_z_acceleration": "frame_acc"}, 
                         inplace=True)
        
        # angle data is not needed for track, blinker is not used anywhere
        framedata.drop(columns=['frame_angle', 'frame_blinker', 'frame_x_position',
                                'frame_state','ballvelocity_first_package',
                                'ballvelocity_last_package'], 
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


def get_BehaviorPose(session_fullfname, ):
    L = Logger()
    nas_dir, _, _ = aC.device_paths()
    dlc_project_path = os.path.join(nas_dir, aC.DLC_MODEL_SUBDIR)
    dlc_config_path = os.path.join(dlc_project_path, "config.yaml")
    
    # if calculted on different machine, update project base path here to mappend NAS location
    with open(dlc_config_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
        L.logger.debug("Loaded DLC config.yaml content:")
        L.logger.debug(L.fmtmsg(yaml_content))
    if yaml_content.get('project_path') != dlc_project_path:
        L.logger.info(f"Updating DLC config path to {dlc_project_path}")   
        yaml_content['project_path'] = dlc_project_path
        with open(dlc_config_path, 'w') as file:
            yaml.safe_dump(yaml_content, file)

    session_dir = os.path.dirname(session_fullfname)
    camera_dir = os.path.join(session_dir, 'rendered_videos')
    CAM = aC.DLC_CAMERA_FNAME.replace('.mp4','')

    # check if there is already camera file mp4 for analysis
    if aC.DLC_CAMERA_FNAME not in [file for file in os.listdir(camera_dir) if file.endswith('.mp4')]:
        L.logger.info(f"Generating {aC.DLC_CAMERA_FNAME} for DLC analysis...")
        result = hdf5_frames2mp4_gpu(session_fullfname, gpu_id=0, camera_names=[CAM], )
        if not result[CAM]:
            # failed to generate video, cannot run DLC analysis, return None
            return
    
    deeplabcut.analyze_videos(
        config=dlc_config_path,
        videos=os.path.join(session_dir, 'rendered_videos', aC.DLC_CAMERA_FNAME),
        videotype='mp4',  # Replace with the video file format if different
        batchsize=aC.DLC_BATCH_SIZE,
        save_as_csv=True,  # Save as intermediate CSVs for inspection
    )
    dlc_result_fname = [f for f in os.listdir(camera_dir) if f.endswith('.csv')][0]
    # read the csv file and reformat
    df_pose = pd.read_csv(os.path.join(camera_dir, dlc_result_fname), low_memory=False)    
    
    # unpack properly, first two rows are multiindex for body part and coordinate
    # combine them and drop scorer level
    df_pose.drop(columns=['scorer'], inplace=True)
    df_pose.columns = df_pose.iloc[0] + '_' + df_pose.iloc[1]
    df_pose = df_pose.iloc[2:]
    df_pose.reset_index(drop=True, inplace=True)
    df_pose = df_pose.astype(float)
    
    # remove dlc files again
    dlc_files = [f for f in os.listdir(camera_dir) if not f.endswith('.mp4')]
    for f in dlc_files:
        os.remove(os.path.join(camera_dir, f))
    
    timestamps = session_modality_from_nas(session_fullfname, f"{CAM}_packages")
    # reindex to full range of image ids, can be missing
    timestamps = timestamps.set_index(f"{CAM}_image_id").reindex(range(timestamps.index.min(), 
                                                                       timestamps.index.max() + 1))  
    # interpolate missing timestamps linearly, if any
    timestamps[f'{CAM}_image_pc_timestamp'] = timestamps[f'{CAM}_image_pc_timestamp'].interpolate(method='linear')
    timestamps[f'{CAM}_image_ephys_timestamp'] = timestamps[f'{CAM}_image_ephys_timestamp'].interpolate(method='linear')

    length_mism = len(df_pose) - len(timestamps)
    if length_mism != 0:
        L.logger.error(f"Length mismatch between DLC pose data and timestamps: {len(df_pose)} vs {len(timestamps)}. Diff: {length_mism}. ")
        return
    
    us_timestamps = timestamps[f'{CAM}_image_ephys_timestamp']
    if pd.isna(us_timestamps).all():
        us_timestamps = timestamps[f'{CAM}_image_pc_timestamp']
    
    # add kinematic features, this will also filter out low-likelihood frames and 
    # frames with high angular velocity (potentially tracking errors)
    df_pose = add_kinematic_features(
        df=df_pose,
        skeleton=yaml_content.get('skeleton'),
        timestamps_us=us_timestamps.values,
        min_likelihood=aC.DLC_MIN_LIKELIHOOD,
        max_angular_vel=aC.DLC_MAX_ANGULAR_VELOCITY,
        segment_names=aC.DLC_SKELETON_NAMES,
    )
    
    # add back the timestamps for merging with framewise data
    df_pose["image_pc_timestamp"] = timestamps[f"{CAM}_image_pc_timestamp"].values
    df_pose["image_ephys_timestamp"] = timestamps[f"{CAM}_image_ephys_timestamp"].values
    return df_pose