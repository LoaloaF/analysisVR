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

            "trial_id": 'first',
            "cue": 'first',
            "trial_outcome": 'first',
            "choice_R1": 'first',
            "choice_R2": 'first',
            "trial_start_pc_timestamp": 'first',

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
            "track_zone": 'first',
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
        
        # seeing which cue
        frame_visible = (behavior_aligned['frame_position'] > -120) & (behavior_aligned['frame_position'] < 25)
        # 0: no cue visible, 1: cue1 visible, 2: cue2 visible
        df['visible_cue'] = frame_visible.astype(float)
        df['visible_cue'][(df['visible_cue'] == 1) & (behavior_aligned['cue'] == 2)] = 2
        
        # Velocity and raw movement
        df['frame_velocity'] = df['frame_velocity'].clip(0, 200)
        # 4 abs_frame_raw quantile binning
        df['frame_raw_quantile'] = pd.qcut(df['frame_raw'], q=4, labels=False, duplicates='drop')

        # Acceleration components
        df['frame_acceleration'] = df['frame_acceleration'].clip(-100, 100)
        df['abs_frame_acceleration'] = df['frame_acceleration'].abs()
        df['frame_positive_acceleration'] = df['frame_acceleration'].clip(0,100)
        df['frame_negative_acceleration'] = df['frame_acceleration'].clip(-100,0)

        # Yaw components
        df['abs_frame_yaw'] = df['frame_yaw'].abs()
        df['frame_yaw_left'] = df['frame_yaw'].clip(-1e6, 0)
        df['frame_yaw_right'] = df['frame_yaw'].clip(0, 1e6)

        # Pitch components
        df['abs_frame_pitch'] = df['frame_pitch'].abs()
        df['frame_pitch_left'] = df['frame_pitch'].clip(-1e6, 0)
        df['frame_pitch_right'] = df['frame_pitch'].clip(0, 1e6)
        
        # Position clipping
        iti_mask = (df['frame_position'] > 260) | (df['frame_position'] < -165)
        df.loc[iti_mask, 'frame_position'] = np.nan
        df['track_zone'] = behavior_aligned['track_zone']

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
            'trial_id',
            'cue',
            'trial_outcome',
            'choice_R1',
            'choice_R2',
            'trial_start_pc_timestamp',
            'from_ephys_timestamp',
            'to_ephys_timestamp',
            
            # forward velocity related
            'frame_velocity',
            'frame_raw',
            'frame_raw_quantile',
            
            # acceleration related
            'frame_acceleration',
            'abs_frame_acceleration',
            'frame_positive_acceleration',
            'frame_negative_acceleration',

            # yaw related
            'frame_yaw',
            'frame_yaw_left',
            'frame_yaw_right',

            # pitch related
            'frame_pitch',
            'frame_pitch_left',
            'frame_pitch_right',
            
            'lick_count',
            'post_lick',
            'post_reward_sound',
            'post_reward',
            
            # zone_before_reward1	zone_before_reward2	zone_between_cues	zone_cue2	zone_cue2_passed	zone_cue2_visible	zone_post_reward	zone_reward1	zone_reward2
            'frame_position',
            'visible_cue',
            # 'track_zone',
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
    behavior['track_zone'] = behavior['track_zone'].cat.add_categories(['ITI'])
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

def get_TrialWiseT0Events40ms(behavior):
    def select_t0_events(trial_beh):
        
        if trial_beh['trial_id'].iloc[0] == -1:
            return pd.DataFrame()  # skip ITI
        base_info = trial_beh[['cue', 'trial_outcome', 'choice_R1', 'choice_R2']].iloc[0].to_dict()

        zone_ts = []
        for zone_name, zone_x in zones_boundaries.items():
            t = np.abs(trial_beh['frame_position']-zone_x).sort_values()
            # print(zone_name)
            if not t.iloc[0]<3: # a valid location should be within 3 cm of the zone
                print(f"Warning: No t0 event found for {zone_name}, was {t.iloc[0]} cm away")
                continue

            zone_ts.append({**base_info, 
                            't0_event_name': zone_name, 
                            't0': t.index[0], 
                            'x_position': trial_beh.loc[t.index[0], 'frame_position'],
                            'x_alignment': zone_x})
        zone_ts = pd.DataFrame(zone_ts)
        return zone_ts

    zones_boundaries = {
        'cueZone_visible': -120,
        'cueZone_entry': -80,
        'cueZone_exit': 25,
        'enter_reward1Zone': 50,
        'enter_reward2Zone': 170,
    }
    
    # kinematics
    # TODO find t points of specific acceleration events, and deceleration events
    # acc = sess_behavior['frame_acceleration']
    # acc.loc[:] = np.clip(acc, -15, 15)

    # TODO find t points of specific acceleration events, and deceleration events also here
    # # shift down the ephys index by one second, predict future acceleration
    # forward_acc_in1sec = acc.copy()
    # nbins_1sec = int(1 * 1_000_000 / bin_size_us)
    # idx = forward_acc_in1sec.index.values[:-nbins_1sec]
    # forward_acc_in1sec = forward_acc_in1sec.iloc[nbins_1sec:]
    # forward_acc_in1sec.index = idx
    
    interval_specifier = {
        # for decoding before cue vs in cue, sepeartely for cue1 and cue2
        'pre_cue_interval': {'zone_alignment': 'cueZone_visible', 'n_bins_left': 5, 'n_bins_right': 0},
        'nextto_cue_interval': {'zone_alignment': 'cueZone_entry', 'n_bins_left': 0, 'n_bins_right': 5},
        # for decoding cue, outcome, choice - early in the track
        'cue_entry_interval': {'zone_alignment': 'cueZone_visible', 'n_bins_left': 3, 'n_bins_right': 37},
        'cue_exit_interval': {'zone_alignment': 'cueZone_exit', 'n_bins_left': 10, 'n_bins_right': 30},
        'R1_entry_interval': {'zone_alignment': 'enter_reward1Zone', 'n_bins_left': 10, 'n_bins_right': 30},
        'R2_entry_interval': {'zone_alignment': 'enter_reward2Zone', 'n_bins_left': 10, 'n_bins_right': 30},
    }
    
    # make a table where every row is a t0 event, keep info like cue, outcome, choice, trial_id
    behavior.set_index('from_ephys_timestamp', inplace=True)
    t0_events = behavior.groupby('trial_id').apply(select_t0_events, 
                                    include_groups=True).reset_index().drop(columns='level_1')

    bin_length_us = 40_000  # 40 ms bins
    
    # instead create intervals based on specifier dict
    for interval_name, spec in interval_specifier.items():
        zone_align = spec['zone_alignment']
        n_bins_left = spec['n_bins_left']
        n_bins_right = spec['n_bins_right']
        
        t0s = t0_events[t0_events.t0_event_name == zone_align]
        
        lefts = t0s.t0 - (n_bins_left * bin_length_us)
        rights = t0s.t0 + (n_bins_right * bin_length_us)
        
        t0_events.loc[t0s.index, interval_name] = pd.IntervalIndex.from_arrays(
            left=lefts,
            right=rights,
            closed='right'
        )
        
        # Create bin array for each row that matches t0s
        bins = pd.Series([np.arange(-n_bins_left, n_bins_right).tolist()] * len(t0s.index))
        t0_events.loc[t0s.index, interval_name+"_bins"] = bins
    return t0_events