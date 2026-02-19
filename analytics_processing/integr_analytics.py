import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger

import analytics_processing.modality_transformations as mT
import analytics_processing.analytics_transformations as aT

def get_BehaviorFramewise(track_kinematics, trialwise, events, pose_data):
    which_t_col = '_ephys_timestamp' 
    if track_kinematics.frame_ephys_timestamp.isna().any():
        Logger().logger.warning("No ephys timestamps in track kinematics, using "
                                "PC timestamps for merging behavior events.")
        which_t_col = '_pc_timestamp'
        
    # below threshold column
    vel_kin = track_kinematics[['frame_RawYawPitch_abs_vel_sum', 'fps', 'track_zone', 'trial_id']]
    thresholds = trialwise[['velocity_threshold_at_R1', 'velocity_threshold_at_R2', 'trial_id']]
    
    below_vel_thr = mT.calculate_below_vel_thr(vel_kin, thresholds)
    track_kinematics['frame_below_velocity_thr'] = below_vel_thr
    
    # transform the async events to framewise data, needed for merging
    fr_wise_events = aT.async_events_to_framewise(events, which_t_col=which_t_col,
                                                  track_kinematics=track_kinematics)
    # merge events in
    framedata = pd.merge(track_kinematics, fr_wise_events, on='frame_id', how='left')
    
    # fill NaNs with 0 for *event*_count columns, mean value stays NaN if 0 events
    cnt_cols = framedata.columns.str.endswith('_count')
    framedata.loc[:, cnt_cols] = framedata.loc[:, cnt_cols].fillna(0)
    
    # create boolean columns for whether an event was detected in that frame
    event_detected = (framedata.loc[:, cnt_cols].copy() > 0).astype(bool).astype(int)
    event_detected.columns= [col.replace('_count', '_detected') for col in event_detected.columns]
    # licks are often flipping (0,1,1,0,1,1), messing up hmm fits, dilate 5 frames (200 ms at 60fps) into future
    event_detected['lick_detected'] = mT.dilate_post_events(event_detected['lick_detected'], n_bins_after=5)
    
    framedata = pd.concat([framedata, event_detected], axis=1)
    
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

def get_BehaviorTrackwise(framewise):
    groupby_cols = ['trial_id', 'from_cm_position_bin', 'to_cm_position_bin']
    # aggregate the data for unque combinations of trial_id and position bin
    posbin_data = framewise.groupby(groupby_cols, observed=True).agg(aT.column2agg_map(framewise.columns))
    
    # by default in agg, we take the last timestamp of ephys_ and pc_timestamp, clarify here by renaming
    posbin_data.rename({'frame_ephys_timestamp': 'posbin_to_ephys_timestamp',
                        'frame_pc_timestamp': 'posbin_to_pc_timestamp'}, inplace=True, axis=1)
    # also add the first timestamp
    first_tstamps = framewise.groupby(groupby_cols, observed=True).agg({
        "frame_ephys_timestamp": 'first',
        "frame_pc_timestamp": 'first',
    }).rename({
        "frame_ephys_timestamp": "posbin_from_ephys_timestamp",
        "frame_pc_timestamp": "posbin_from_pc_timestamp",
    }, axis=1)
    posbin_data = pd.concat([posbin_data, first_tstamps], axis=1)
    # print(posbin_data)
    # print(posbin_data[['posbin_from_ephys_timestamp','posbin_from_pc_timestamp']])
    
    # add a column with the number of frames in each position bin
    posbin_data['nframes_in_posbin'] = framewise.groupby(groupby_cols, observed=False).size()
    # replace frame_ with posbin_ in the column names
    renamer = {col: col.replace('frame_', 'posbin_') for col in posbin_data.columns}
    posbin_data.rename(columns=renamer, inplace=True)
    
    posbin_data = posbin_data.groupby(level='trial_id').apply(aT.interp_missing_pos_bins)
    posbin_data.reset_index(level=[0,1], drop=True, inplace=True) # drop trial_id, already have it
    # from_cm_position_bin, to_cm_position_bin, trial_id back into df
    posbin_data.reset_index(inplace=True)
    return posbin_data

def get_Behavior40msAligned(fr, behavior):
    # Create interval index for ephys bins
    ephys_intervals = pd.IntervalIndex.from_arrays(
        fr['from_ephys_timestamp'],
        fr['to_ephys_timestamp'],
        closed='right'
    )
    
    # Assign each frame to an ephys bin
    frame_bin_assignment = pd.cut(
        behavior['frame_ephys_timestamp'],
        bins=ephys_intervals,
        labels=np.arange(len(ephys_intervals))
    )
    
    # Add bin assignment back to frame data
    behavior['ephys_bin_id'] = frame_bin_assignment

    # Group frames by ephys bins and aggregate
    behavior_aligned = behavior.groupby('ephys_bin_id').agg(aT.column2agg_map(behavior.columns))
    
    # Add ephys bin timestamps
    behavior_aligned['from_ephys_timestamp'] = fr['from_ephys_timestamp'].values
    behavior_aligned['to_ephys_timestamp'] = fr['to_ephys_timestamp'].values
    return behavior_aligned

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