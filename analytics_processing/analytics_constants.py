from collections import OrderedDict
import pandas as pd

# ALL_PARADIGM_IDS = [200, 800, 1100]
# ALL_ANIMAL_IDS = [1,2,3,4,5,6,7,8,9]
LICK_MERGE_INTERVAL = 0.03

# TODO check dtypes
SESSION_METADATA_TABLE = OrderedDict([
    ("trial_id", pd.Int16Dtype()),
    ("session_name", pd.StringDtype()),
    ("paradigm_name", pd.StringDtype()),
    ("paradigm_id", pd.Int16Dtype()),
    ("animal_name", pd.StringDtype()),
    ("animal_id", pd.Int16Dtype()),
    ("animal_weight", pd.StringDtype()),
    ("start_time", pd.StringDtype()),
    ("stop_time", pd.StringDtype()),
    ("duration_minutes", pd.Float32Dtype()),
    ("notes", pd.StringDtype()),
    ("rewardPostSoundDelay", pd.Float32Dtype()),
    ("rewardAmount", pd.Float32Dtype()),
    ("successSequenceLength", pd.Float32Dtype()),
    ("trialPackageVariables", pd.StringDtype()),
    ("trialPackageVariablesDefault", pd.StringDtype()),
    ("track_details", pd.StringDtype()),
    
    ("GAP", pd.Float32Dtype()), # random keys from here on
    ("session_id", pd.StringDtype()),
    ("duration", pd.StringDtype()),
    ("punishmentLength", pd.Float32Dtype()),
    ("punishmentInactivationLength", pd.Float32Dtype()),
    ("onWallZoneEntry", pd.StringDtype()),
    ("onInterTrialInterval", pd.StringDtype()),
    ("interTrialIntervalLength", pd.Float32Dtype()),
    ("abortInterTrialIntervalLength", pd.Float32Dtype()),
    ("maxiumTrialLength", pd.Float32Dtype()),
    ("sessionFREEVAR2", pd.StringDtype()),
    ("sessionDescription", pd.StringDtype()),
    ("sessionFREEVAR4", pd.StringDtype()),
    ("trialPackageVariablesFulllNames", pd.StringDtype()),
    ("trialPackageVariablesFullNames", pd.StringDtype()),
    ("metadata", pd.StringDtype()),
])

# Define the schema with pandas dtypes
UNITY_FAMEWISE_TABLE = OrderedDict([
    ("frame_x_position", pd.Float32Dtype()),
    ("frame_angle", pd.Float32Dtype()),
    
    ("lick_reward", pd.BooleanDtype()),
    ("maximum_reward_number", pd.Int32Dtype()),
    ("stay_time", pd.Float32Dtype()),
    ("cue", pd.Int16Dtype()),
    
    ("trial_id", pd.Int16Dtype()),
    ("trial_start_frame", pd.Int32Dtype()),
    ("trial_start_pc_timestamp", pd.Int64Dtype()),
    ("trial_end_frame", pd.Int32Dtype()),
    ("trial_end_pc_timestamp", pd.Int64Dtype()),
    ("trial_pc_duration", pd.Int32Dtype()),
    ("trial_outcome", pd.Int32Dtype()),
    ("trial_start_ephys_timestamp", pd.Int64Dtype()),
    ("trial_end_ephys_timestamp", pd.Int64Dtype()),
    
    ("zone", pd.StringDtype()),
    ("frame_state", pd.Int16Dtype()),
    
    ("from_z_position_bin", pd.Int16Dtype()),  # Nullable integer
    ("to_z_position_bin", pd.Int16Dtype()),    # Nullable integer
    ("frame_z_position", pd.Float32Dtype()),
    ("frame_z_velocity", pd.Float32Dtype()),  # Nullable float
    ("frame_z_acceleration", pd.Float32Dtype()), # Nullable float
    ("frame_id", pd.Int64Dtype()),
    ("frame_pc_timestamp", pd.Int64Dtype()),
    ("frame_ephys_timestamp", pd.Int64Dtype()), # Nullable integer
    ("ballvelocity_first_package", pd.Int64Dtype()),
    ("ballvelocity_last_package", pd.Int64Dtype()),
    ("frame_blinker", pd.BooleanDtype()),      # Nullable boolean
    
])

# UNITY_TRIALWISE_TABLE = OrderedDict([
#     ("stay_time", pd.Float32Dtype()),                      #   1
#     ("maximum_reward_number", pd.Int16Dtype()),          #   1
#     ("lick_reward", pd.BooleanDtype()),                    #   1
#     ("cue", pd.Int8Dtype()),                            #   2
#     ("trial_outcome", pd.Int8Dtype()),                  #   6
    
#     ("trial_id", pd.Int16Dtype()),                       # 251
#     ("trial_start_frame", pd.Int32Dtype()),              # 251
#     ("trial_end_frame", pd.Int32Dtype()),                # 251
#     ("trial_start_pc_timestamp", pd.Int64Dtype()),       # 251
#     ("trial_end_pc_timestamp", pd.Int64Dtype()),         # 251
#     ("trial_pc_duration", pd.Int64Dtype()),              # 251
#     ("trial_start_ephys_timestamp", pd.Float64Dtype()),    #   0
#     ("trial_end_ephys_timestamp", pd.Float64Dtype()),      #   0
#     ("enter_reward1", pd.Int64Dtype()),                  # 251
#     ("enter_reward2", pd.Int64Dtype()),                  # 251
#     ("staytime_before_reward1", pd.Int64Dtype()),        # 251
#     ("staytime_before_reward2", pd.Int64Dtype()),        # 251
#     ("staytime_between_cues", pd.Int64Dtype()),          # 250
#     ("staytime_correct_r", pd.Int64Dtype()),             # 251
#     ("staytime_cue1", pd.Int64Dtype()),                  # 249
#     ("staytime_cue1_passed", pd.Int64Dtype()),           # 249
#     ("staytime_cue1_visible", pd.Int64Dtype()),          #   1
#     ("staytime_cue2", pd.Int64Dtype()),                  # 248
#     ("staytime_cue2_passed", pd.Int64Dtype()),           # 251
#     ("staytime_cue2_visible", pd.Int64Dtype()),          # 248
#     ("staytime_incorrect_r", pd.Int64Dtype()),           # 251
#     ("staytime_post_reward", pd.Int64Dtype()),           # 250
#     ("staytime_reward1", pd.Int64Dtype()),               # 251
#     ("staytime_reward2", pd.Int64Dtype()),               # 251
#     ("staytime_start_zone", pd.Int64Dtype()),            #   1
# ])

POSITION_BIN_TABLE_EXCLUDE = ["frame_x_position", "frame_angle", 
                              "ballvelocity_first_package", 
                              "ballvelocity_last_package", "frame_blinker"]

UNITY_TRACKWISE_TABLE = OrderedDict([
    ("lick_reward", pd.BooleanDtype()),
    ("maximum_reward_number", pd.Int32Dtype()),
    ("stay_time", pd.Float32Dtype()),
    ("cue", pd.Int16Dtype()),
    
    ("trial_id", pd.Int16Dtype()),
    ("trial_start_frame", pd.Int32Dtype()),
    ("trial_start_pc_timestamp", pd.Int64Dtype()),
    ("trial_end_frame", pd.Int32Dtype()),
    ("trial_end_pc_timestamp", pd.Int64Dtype()),
    ("trial_pc_duration", pd.Int32Dtype()),
    ("trial_outcome", pd.Int32Dtype()),
    ("trial_start_ephys_timestamp", pd.Int64Dtype()),
    ("trial_end_ephys_timestamp", pd.Int64Dtype()),
    
    ("zone", pd.StringDtype()),
    ("posbin_state", pd.Int16Dtype()),
    
    ("from_z_position_bin", pd.Int16Dtype()),
    ("to_z_position_bin", pd.Int16Dtype()),
    
    ("nframes_in_bin", pd.Int16Dtype()),
    
    ("posbin_z_position", pd.Float32Dtype()),
    ("posbin_z_velocity", pd.Float32Dtype()), # Nullable float
    ("posbin_z_acceleration", pd.Float32Dtype()), # Nullable float
    
    ("posbin_from_frame_id", pd.Int64Dtype()),
    ("posbin_from_pc_timestamp", pd.Int64Dtype()),
    ("posbin_from_ephys_timestamp", pd.Int64Dtype()), # Nullable integer
    ("posbin_to_posbin_id", pd.Int64Dtype()),
    ("posbin_to_pc_timestamp", pd.Int64Dtype()),
    ("posbin_to_ephys_timestamp", pd.Int64Dtype()), # Nullable integer
])
    