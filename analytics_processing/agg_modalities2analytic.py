# import numpy as np
import json
import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

import analytics_processing.modality_transformations as mT

from analytics_processing.modality_loading import session_modality_from_nas

def get_SesssionMetadata(session_fullfname):
    data = session_modality_from_nas(session_fullfname, "metadata")
    # drop nested json-like fields
    data = {k:v for k,v in data.items() 
            if k not in ["env_metadata","fsm_metadata","configuration","log_file_content"]}
    data = pd.Series(data, name=0).to_frame().T
    return data

# TDOO: add pose parquet generation
# Take into facecam images and fed into pre-trained model

def get_BehaviorEvents(session_fullfname):
    eventdata = session_modality_from_nas(session_fullfname, "event")
    balldata = session_modality_from_nas(session_fullfname, "ballvelocity")
    
    # process event
    eventdata.drop(columns=['event_portenta_timestamp', 'trial_id'], inplace=True)
    # adjus the lick start time
    lick_value = eventdata[eventdata["event_name"] == "L"]["event_value"]
    eventdata.loc[eventdata["event_name"] == "L", "event_pc_timestamp"] += lick_value
    if not eventdata["event_ephys_timestamp"].isna().any():
        eventdata.loc[eventdata["event_name"] == "L", "event_ephys_timestamp"] += lick_value
        eventdata.loc[eventdata["event_name"] == "L", "event_ephys_timestamp"] = round(eventdata.loc[eventdata["event_name"] == "L", "event_ephys_timestamp"]/50) * 50
    eventdata.loc[eventdata["event_name"] == "L", "event_value"] *= -1
    # check the patched indicator
    if "event_ephys_patched" not in eventdata.columns:
        eventdata["event_ephys_patched"] = np.nan
    
    # process ball velocity
    balldata.drop(columns=['ballvelocity_portenta_timestamp', 'trial_id'], inplace=True)
    balldata["event_name"] = "B"
    balldata["event_value"] = balldata["ballvelocity_raw"].astype(str) + "," + balldata["ballvelocity_yaw"].astype(str) + "," + balldata["ballvelocity_pitch"].astype(str)
    balldata.drop(columns=['ballvelocity_raw', 'ballvelocity_yaw', 'ballvelocity_pitch'], inplace=True)
    balldata.rename(columns={'ballvelocity_pc_timestamp': 'event_pc_timestamp', 
                             'ballvelocity_ephys_timestamp': 'event_ephys_timestamp',
                             'ballvelocity_package_id': 'event_package_id'}, inplace=True)
    if "ballvelocity_ephys_patched" in balldata.columns:
        balldata.rename(columns={'ballvelocity_ephys_patched': 'event_ephys_patched'}, inplace=True)
    else:
        balldata["event_ephys_patched"] = np.nan
    
    behavior_events = pd.concat([eventdata, balldata], ignore_index=True)
    behavior_events.reset_index(drop=True, inplace=True)
    return behavior_events
    # event_name; event_package_id; event_pc; event_ephys; event_value(string);
    # ball_sensor; Lick; Sound; Reward; Sucktion
    
    # raise NotImplementedError
    # #TODO process licks poerperly, make reward timeline with vacuum
    # #TODO add pose estimation
    # events_data = session_modality_from_nas(session_fullfname, "events")
    # ballvel_data = session_modality_from_nas(session_fullfname, "ballvelocity")
    # facecam_data = session_modality_from_nas(session_fullfname, "facecam")
    
    # lickdata = data.loc[data['event_name'] == "L"]
    # licks = mT.event_modality_calc_timeintervals_around_lick(lickdata, C.LICK_MERGE_INTERVAL)
    
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
        print(staytimes)
        trialdata = pd.concat([trialdata, staytimes], axis=1)
    print(trialdata)
    
    return trialdata