import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger

import analytics_processing.analytics_transformations as aT

def get_BehaviorFramewise(track_kinematics, trialwise, events, pose_data):
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
    timestamp_col = 'frame_ephys_timestamp'
    if framedata.loc[:,timestamp_col].isna().iloc[0]:
        timestamp_col = 'frame_pc_timestamp'
    camera_timestamp_col = timestamp_col.replace('frame', 'facecam_image')

    # Create a copy of pose_data with timestamp as index for merging
    pose_data_indexed = pose_data.set_index(camera_timestamp_col)
    
    # Create a Series mapping frame timestamps to nearest camera timestamps
    frame_to_camera = pd.Series(index=framedata[timestamp_col], 
                               data=framedata[timestamp_col].values).map(
        lambda x: pose_data_indexed.index[abs(pose_data_indexed.index - x).argmin()]
    )
                               
    # Use the mapping to merge pose data
    pose_data_matched = pose_data_indexed.loc[frame_to_camera].reset_index()
    pose_data_matched.index = framedata.index
    
    # Merge the matched pose data with framedata
    framedata = pd.concat([framedata, pose_data_matched.drop(columns=[camera_timestamp_col])], axis=1)
    
    # merge trialwise data in (big)
    framedata = pd.merge(framedata, trialwise, on='trial_id', how='left')
    return framedata

def get_BehaviorTrackwise(tramewise):
    return aT.transform_to_position_bin_index(tramewise)

def get_Behavior40msAligned(fr, behavior):
    def align_frame_to_ephys_bins(session_ephys, session_behavior):
        # Create interval index for ephys bins
        ephys_intervals = pd.IntervalIndex.from_arrays(
            session_ephys['from_ephys_timestamp'],
            session_ephys['to_ephys_timestamp'],
            closed='right'
        )
        
        # Assign each frame to an ephys bin
        frame_bin_assignment = pd.cut(
            session_behavior['frame_ephys_timestamp'],
            bins=ephys_intervals,
            labels=np.arange(len(ephys_intervals))
        )
        
        # Add bin assignment back to frame data
        session_behavior = session_behavior.copy()
        session_behavior['ephys_bin_id'] = frame_bin_assignment
        
        # Group frames by ephys bins and aggregate
        grouped_data = session_behavior.groupby('ephys_bin_id').agg({
            "frame_velocity": 'mean',
            "frame_acceleration": 'mean',
            "frame_raw": 'mean',
            "frame_yaw": 'mean',
            "frame_pitch": 'mean',
            "lick_count": 'sum',
            
            "facecam_pose_nose_neck_body1_angle": 'mean',
            "facecam_pose_nose_neck_body1_angle_likelihood": 'mean',
            "facecam_pose_nose_neck_body1_angle_velocity": 'mean',
            
            "zone_before_reward1": 'mean',
            "zone_before_reward2": 'mean',
            "zone_between_cues": 'mean',
            "zone_cue2": 'mean',
            "zone_cue2_passed": 'mean',
            "zone_cue2_visible": 'mean',
            "zone_post_reward": 'mean',
            "zone_reward1": 'mean',
            "zone_reward2": 'mean',
            
            # state based
            "frame_position": 'mean',
            # "reward-removed_count": 'sum',
            "reward-sound_count": 'sum',
            "reward-valve-open_count": 'sum',
        })
        
        # Add ephys bin timestamps
        grouped_data['from_ephys_timestamp'] = session_ephys['from_ephys_timestamp'].values
        grouped_data['to_ephys_timestamp'] = session_ephys['to_ephys_timestamp'].values

        # zone columns back to bool type
        zone_cols = [col for col in grouped_data.columns if col.startswith('zone_')]
        for col in zone_cols:
            grouped_data[col] = grouped_data[col] > 0.5
        
        return grouped_data

    def dilate_post_events(event_timeseries, n_bins_after=4):
        dilated = event_timeseries.copy()
        # Find indices where events occur
        event_indices = np.where(event_timeseries == 1)[0]
        # For each event, set the next n_bins to 1
        for idx in event_indices:
            dilated[idx:min(idx + n_bins_after + 1, len(event_timeseries))] = 1
        return dilated
    
    # Clean up behavior variables with proper pandas operations
    def preprocess_behavior(behavior_aligned):
        """Clean and preprocess behavior variables."""
        df = behavior_aligned.copy()
        
        # Velocity and raw movement
        df['frame_velocity'] = df['frame_velocity'].clip(0, 200)
        df['abs_frame_raw'] = df['frame_raw'].abs()
        # 4 abs_frame_raw quantile binning
        df['abs_frame_raw_quantile'] = pd.qcut(df['abs_frame_raw'], q=4, labels=False, duplicates='drop')
        
        # Acceleration components
        df['frame_acceleration'] = df['frame_acceleration'].clip(-100, 100)
        df['abs_frame_acceleration'] = df['frame_acceleration'].abs()
        df['frame_decceleration'] = df['frame_acceleration'].copy()
        df.loc[df['frame_decceleration'] > 0, 'frame_decceleration'] = np.nan
        df.loc[df['frame_decceleration'] <= 0, 'frame_acceleration'] = np.nan
        df['frame_decceleration'] *= -1  # make decceleration positive
        
        # Yaw components
        df['abs_frame_yaw'] = df['frame_yaw'].abs()
        df['frame_yaw_left'] = df['frame_yaw'].copy()
        df['frame_yaw_right'] = df['frame_yaw'].copy()
        df.loc[df['frame_yaw'] > 0, 'frame_yaw_left'] = np.nan
        df.loc[df['frame_yaw'] <= 0, 'frame_yaw_right'] = np.nan
        df['frame_yaw_left'] *= -1  # make left yaw positive
        
        # Pitch components
        df['abs_frame_pitch'] = df['frame_pitch'].abs()
        df['frame_pitch_left'] = df['frame_pitch'].copy() 
        df['frame_pitch_right'] = df['frame_pitch'].copy()
        df.loc[df['frame_pitch'] > 0, 'frame_pitch_left'] = np.nan
        df.loc[df['frame_pitch'] <= 0, 'frame_pitch_right'] = np.nan
        df['frame_pitch_left'] *= -1  # make left pitch positive
        
        # Position clipping
        iti_mask = (df['frame_position'] > 260) | (df['frame_position'] < -160)
        df.loc[iti_mask, 'frame_position'] = np.nan

        df['lick_count'] = df['lick_count'].clip(0, 1)
        df['post_lick'] = dilate_post_events(df['lick_count'].values, n_bins_after=2)
        df['post_reward_sound'] = dilate_post_events(df['reward-sound_count'].values, n_bins_after=6)
        df['post_reward'] = dilate_post_events(df['reward-valve-open_count'].values, n_bins_after=6)
        
        
        # apply likelihood thresholding to angle and angle velocity
        thr = .5
        angle_mask = behavior_aligned[f"facecam_pose_nose_neck_body1_angle_likelihood"] > thr
        df['head_angle'] = behavior_aligned['facecam_pose_nose_neck_body1_angle']
        df.loc[~angle_mask, 'head_angle'] = np.nan

        # left right split
        head_angle_left_mask = df['facecam_pose_nose_neck_body1_angle'] < df['facecam_pose_nose_neck_body1_angle'].median()
        df['head_angle_left'] = behavior_aligned['facecam_pose_nose_neck_body1_angle']
        df.loc[~head_angle_left_mask, 'head_angle_left'] = np.nan
        df['head_angle_right'] = behavior_aligned['facecam_pose_nose_neck_body1_angle']
        df.loc[head_angle_left_mask, 'head_angle_right'] = np.nan

        df['head_angle_velocity'] = behavior_aligned['facecam_pose_nose_neck_body1_angle_velocity']
        df.loc[~angle_mask, 'head_angle_velocity'] = np.nan
        
        # left right
        head_angle_left_mask = df['facecam_pose_nose_neck_body1_angle_velocity'] < df['facecam_pose_nose_neck_body1_angle_velocity'].median()
        df['head_angle_velocity_left'] = behavior_aligned['facecam_pose_nose_neck_body1_angle_velocity']
        df.loc[~head_angle_left_mask, 'head_angle_velocity_left'] = np.nan
        df['head_angle_velocity_right'] = behavior_aligned['facecam_pose_nose_neck_body1_angle_velocity']
        df.loc[head_angle_left_mask, 'head_angle_velocity_right'] = np.nan

        
        df['zone_before_reward1'] = behavior_aligned['zone_before_reward1'].astype(float)
        df['zone_before_reward2'] = behavior_aligned['zone_before_reward2'].astype(float)
        df['zone_between_cues'] = behavior_aligned['zone_between_cues'].astype(float)
        df['zone_cue2'] = behavior_aligned['zone_cue2'].astype(float)
        df['zone_cue2_passed'] = behavior_aligned['zone_cue2_passed'].astype(float)
        df['zone_cue2_visible'] = behavior_aligned['zone_cue2_visible'].astype(float)
        df['zone_post_reward'] = behavior_aligned['zone_post_reward'].astype(float)
        df['zone_reward1'] = behavior_aligned['zone_reward1'].astype(float)
        df['zone_reward2'] = behavior_aligned['zone_reward2'].astype(float)

        # Reorder columns
        column_order = [
            'frame_velocity',
            'abs_frame_raw_quantile',
            'abs_frame_raw',
            'frame_acceleration',
            'frame_decceleration', 
            'abs_frame_acceleration',
            'frame_yaw_left',
            'frame_yaw_right',
            'abs_frame_yaw',
            'abs_frame_pitch',
            'frame_pitch_left',
            'frame_pitch_right',
            # 'lick_count',
            # 'reward-sound_count',
            # 'reward-valve-open_count'
            'lick_count',
            # 'post_lick',
            'post_reward_sound',
            'post_reward',
            
            # zone_before_reward1	zone_before_reward2	zone_between_cues	zone_cue2	zone_cue2_passed	zone_cue2_visible	zone_post_reward	zone_reward1	zone_reward2
            'frame_position',
            'zone_before_reward1',
            'zone_before_reward2',
            'zone_between_cues',
            
            'zone_cue2',
            'zone_cue2_passed',
            'zone_cue2_visible',
            'zone_post_reward',
            'zone_reward1',
            'zone_reward2',

            'head_angle',
            'head_angle_left',
            'head_angle_right',

            'head_angle_velocity',
            'head_angle_velocity_left',
            'head_angle_velocity_right',
        ]
        
        return df[column_order]

    iti_mask = (behavior['frame_position'] > 260) | (behavior['frame_position'] < -160)
    behavior.loc[iti_mask, 'track_zone'] = 'ITI'

    # add columns for each zone, one hot encoding
    behavior['track_zone'].unique()
    zone_dummies = pd.get_dummies(behavior['track_zone'], prefix='zone')

    # concatenate back to behavior
    behavior = pd.concat([behavior, zone_dummies], axis=1)
    
    behavior_aligned = align_frame_to_ephys_bins(fr, behavior)
        
    # Apply preprocessing
    behavior_aligned_cleaned = preprocess_behavior(behavior_aligned).reset_index(drop=True)
    return behavior_aligned_cleaned