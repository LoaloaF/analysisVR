import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger

import analytics_processing.analytics_transformations as aT

def get_BehaviorFramewise(track_kinematics, trialwise, events, ): # pose_data):
    which_t_col = '_ephys_timestamp' 
    if track_kinematics.frame_ephys_timestamp.isna().any():
        Logger().logger.warning("No ephys timestamps in track kinematics, using "
                                "PC timestamps for merging behavior events.")
        which_t_col = '_pc_timestamp'
    
    # transform the async events to framewise data, needed for merging
    fr_wise_events = aT.async_events_to_framewise(events, which_t_col=which_t_col,
                                                  track_kinematics=track_kinematics)
    
    # merge events in
    framedata = pd.merge(track_kinematics, fr_wise_events, on='frame_id', how='left')
    
    # fill NaNs with 0 for *event*_count columns, mean value stays NaN if 0 events
    cnt_cols = framedata.columns.str.endswith('_count')
    framedata.loc[:, cnt_cols] = framedata.loc[:, cnt_cols].fillna(0)
    
    # TODO merge facecam poses with frames
    # framedata = merge_facecam_poses_with_frames(framedata, pose_data)
    
    # merge trialwise data in (big)
    framedata = pd.merge(framedata, trialwise, on='trial_id', how='left')
    return framedata

def get_BehaviorTrackwise(tramewise):
    return aT.transform_to_position_bin_index(tramewise)










