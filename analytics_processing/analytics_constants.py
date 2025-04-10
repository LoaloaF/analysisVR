import os
from collections import OrderedDict
import pandas as pd
import platform 

def device_paths():
    which_os = platform.system()
    user = os.getlogin()
    # print(f"OS: {which_os}, User: {user}")
    
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        project_dir = "/home/houmanjava/meatesting/"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/vrmaster/Projects/VirtualReality/"
    
    elif which_os == 'Linux' and user == 'loaloa':
        nas_dir = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/loaloa/homedataXPS/projects/ratvr/VirtualReality/"
    
    elif which_os == "Darwin" and user == "root":
        nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        # nas_dir = "/Users/loaloa/local_data/nas_imitation"
        folders = [f for f in os.listdir("/Users") if os.path.isdir(os.path.join("/Users", f))]

        if "loaloa" in folders:
            local_data_dir = "/Users/loaloa/local_data/analysisVR_cache"
            project_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/"
        elif "yaohaotian" in folders:
            local_data_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/data/analysisVR_cache"
            project_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/"
        else:
            raise ValueError("Unknown MacOS user")
    
    else:
        nas_dir, local_data_dir, project_dir = None, None, None
        raise ValueError("Unknown OS or user: ", which_os, user)
    
    if not os.path.exists(nas_dir):
        msg = f"NAS directory not found: {nas_dir} - VPN connected?"
        raise FileNotFoundError(msg)
    return nas_dir, local_data_dir, project_dir

# ALL_PARADIGM_IDS = [200, 800, 1100]
# ALL_ANIMAL_IDS = [1,2,3,4,5,6,7,8,9]
LICK_MERGE_INTERVAL = 0.03

MODALITY_KEYS = ['ephys_traces', 'event', 'ballvelocity', 'metadata',
                 'unitycam_packages', 'facecam_packages', 'bodycam_packages',
                 'unity_frame', 'unity_trial', 'paradigm_variable', ]

SESSION_METADATA_TABLE = OrderedDict([
    ("trial_id", pd.Int32Dtype()),
    ("session_name", pd.StringDtype()),
    ("paradigm_name", pd.StringDtype()),
    ("paradigm_id", pd.Int32Dtype()),
    ("animal_name", pd.StringDtype()),
    ("animal_id", pd.Int32Dtype()),
    ("animal_weight", pd.Int32Dtype()),
    ("start_time", pd.StringDtype()),
    ("stop_time", pd.StringDtype()),
    ("duration_minutes", pd.Int32Dtype()),
    ("notes", pd.StringDtype()),
    ("rewardPostSoundDelay", pd.Int32Dtype()),
    ("rewardAmount", pd.Int32Dtype()),
    ("successSequenceLength", pd.Float32Dtype()),
    ("trialPackageVariables", pd.StringDtype()),  # Lists are stored as strings
    ("trialPackageVariablesDefault", pd.StringDtype()),  # Lists are stored as strings
    ("trialPackageVariablesFullNames", pd.StringDtype()),  # Lists are stored as strings
    ("configuration", pd.StringDtype()),  # Dicts are stored as strings
    ("env_metadata", pd.StringDtype()),  # Dicts are stored as strings
    ("fsm_metadata", pd.StringDtype()),  # Dicts are stored as strings
    ("track_details", pd.StringDtype()),
    # ("GAP", pd.StringDtype()),  # NoneType is stored as string
    # ("session_id", pd.StringDtype()),
    # ("duration", pd.StringDtype()),
    # ("punishmentLength", pd.Int32Dtype()),
    # ("punishmentInactivationLength", pd.Int32Dtype()),
    # ("onWallZoneEntry", pd.StringDtype()),
    # ("onInterTrialInterval", pd.StringDtype()),
    # ("interTrialIntervalLength", pd.Int32Dtype()),
    # ("abortInterTrialIntervalLength", pd.Int32Dtype()),
    # ("maxiumTrialLength", pd.Int32Dtype()),
    # ("sessionFREEVAR2", pd.Int32Dtype()),
    # ("sessionDescription", pd.Float32Dtype()),  # NaN is stored as float
    ("ephys_traces_recorded", pd.BooleanDtype()),
    ("ephys_traces_n_columns", pd.Int32Dtype()),
    ("ephys_traces_n_rows", pd.Int32Dtype()),
    # ("ephys_traces_columns", pd.StringDtype()),  # Lists are stored as strings
    
    ("event_n_columns", pd.Int32Dtype()),
    ("event_n_rows", pd.Int32Dtype()),
    ("event_columns", pd.StringDtype()),  # Lists are stored as strings
    ("event_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("event_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("event_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("event_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("ballvelocity_n_columns", pd.Int32Dtype()),
    ("ballvelocity_n_rows", pd.Int32Dtype()),
    ("ballvelocity_columns", pd.StringDtype()),  # Lists are stored as strings
    ("ballvelocity_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("ballvelocity_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("ballvelocity_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("ballvelocity_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("metadata_n_columns", pd.Int32Dtype()),
    ("metadata_n_rows", pd.Int32Dtype()),
    ("metadata_columns", pd.StringDtype()),  # Lists are stored as strings
    ("unitycam_packages_missing_frame_keys", pd.StringDtype()),  # Lists are stored as strings
    ("unitycam_packages_no_missing_frame_keys", pd.BooleanDtype()),
    ("unitycam_packages_n_columns", pd.Int32Dtype()),
    ("unitycam_packages_n_rows", pd.Int32Dtype()),
    ("unitycam_packages_columns", pd.StringDtype()),  # Lists are stored as strings
    ("unitycam_packages_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("unitycam_packages_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("unitycam_packages_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("unitycam_packages_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("facecam_packages_missing_frame_keys", pd.StringDtype()),  # Lists are stored as strings
    ("facecam_packages_no_missing_frame_keys", pd.BooleanDtype()),
    ("facecam_packages_n_columns", pd.Int32Dtype()),
    ("facecam_packages_n_rows", pd.Int32Dtype()),
    ("facecam_packages_columns", pd.StringDtype()),  # Lists are stored as strings
    ("facecam_packages_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("facecam_packages_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("facecam_packages_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("facecam_packages_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("bodycam_packages_missing_frame_keys", pd.StringDtype()),  # Lists are stored as strings
    ("bodycam_packages_no_missing_frame_keys", pd.BooleanDtype()),
    ("bodycam_packages_n_columns", pd.Int32Dtype()),
    ("bodycam_packages_n_rows", pd.Int32Dtype()),
    ("bodycam_packages_columns", pd.StringDtype()),  # Lists are stored as strings
    ("bodycam_packages_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("bodycam_packages_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("bodycam_packages_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("bodycam_packages_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("unity_frame_n_columns", pd.Int32Dtype()),
    ("unity_frame_n_rows", pd.Int32Dtype()),
    ("unity_frame_columns", pd.StringDtype()),  # Lists are stored as strings
    ("unity_frame_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("unity_frame_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("unity_frame_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("unity_frame_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("unity_trial_n_columns", pd.Int32Dtype()),
    ("unity_trial_n_rows", pd.Int32Dtype()),
    ("unity_trial_columns", pd.StringDtype()),  # Lists are stored as strings
    ("unity_trial_n_nan_ephys_timestamp", pd.Int32Dtype()),
    ("unity_trial_no_nan_ephys_timestamps", pd.BooleanDtype()),
    ("unity_trial_n_patched_ephys_timestamp", pd.Int32Dtype()),
    ("unity_trial_no_patched_ephys_timestamps", pd.BooleanDtype()),
    ("paradigm_variable_n_columns", pd.Int32Dtype()),
    ("paradigm_variable_n_rows", pd.Int32Dtype()),
    ("paradigm_variable_columns", pd.StringDtype()),  # Lists are stored as strings
])

# # TODO check dtypes
# SESSION_METADATA_TABLE = OrderedDict([
#     ("trial_id", pd.Int16Dtype()),
#     ("session_name", pd.StringDtype()),
#     ("paradigm_name", pd.StringDtype()),
#     ("paradigm_id", pd.Int16Dtype()),
#     ("animal_name", pd.StringDtype()),
#     ("animal_id", pd.Int16Dtype()),
#     ("animal_weight", pd.StringDtype()),
#     ("start_time", pd.StringDtype()),
#     ("stop_time", pd.StringDtype()),
#     ("duration_minutes", pd.Float32Dtype()),
#     ("notes", pd.StringDtype()),
#     ("rewardPostSoundDelay", pd.Float32Dtype()),
#     ("rewardAmount", pd.Float32Dtype()),
#     ("successSequenceLength", pd.Float32Dtype()),
#     ("trialPackageVariables", pd.StringDtype()),
#     ("trialPackageVariablesDefault", pd.StringDtype()),
#     ("track_details", pd.StringDtype()),
    
#     ("GAP", pd.Float32Dtype()), # random keys from here on
#     ("session_id", pd.StringDtype()),
#     ("duration", pd.StringDtype()),
#     ("punishmentLength", pd.Float32Dtype()),
#     ("punishmentInactivationLength", pd.Float32Dtype()),
#     ("onWallZoneEntry", pd.StringDtype()),
#     ("onInterTrialInterval", pd.StringDtype()),
#     ("interTrialIntervalLength", pd.Float32Dtype()),
#     ("abortInterTrialIntervalLength", pd.Float32Dtype()),
#     ("maxiumTrialLength", pd.Float32Dtype()),
#     ("sessionFREEVAR2", pd.StringDtype()),
#     ("sessionDescription", pd.StringDtype()),
#     ("sessionFREEVAR4", pd.StringDtype()),
#     ("trialPackageVariablesFulllNames", pd.StringDtype()),
#     ("trialPackageVariablesFullNames", pd.StringDtype()),
#     ("metadata", pd.StringDtype()),
# ])

BEHAVIOR_EVENT_TABLE = OrderedDict([
    ("event_name", pd.StringDtype()),
    ("event_package_id", pd.Int64Dtype()),
    ("event_pc_timestamp", pd.Int64Dtype()),
    ("event_ephys_timestamp", pd.Int64Dtype()),
    ("event_value", pd.StringDtype()),
    ("event_ephys_patched", pd.Int8Dtype()),
])

UNITY_TABLE = OrderedDict([
    # compeletly conastant across all paradigms, usful in 2D
    ("frame_x_position", pd.Float32Dtype()),
    ("frame_angle", pd.Float32Dtype()),
    
    # 00% of session constant thoughout session, except cue
    ("lick_triggers_reward", pd.BooleanDtype()),
    ("maximum_reward_number", pd.Int32Dtype()),
    ("stay_time", pd.Float32Dtype()),
    ("multi_reward_requires_stop", pd.BooleanDtype()),
    ("both_R1_R2_rewarded", pd.BooleanDtype()),
    ("flip_Cue1R1_Cue2R2", pd.BooleanDtype()),
    ("prob_cue1_trial", pd.Float32Dtype()),
    ("movement_gain_scaler", pd.Float32Dtype()),
    
    ("cue", pd.Int16Dtype()),
    
    # constant for each frame of a trial
    ("trial_id", pd.Int16Dtype()),
    ("trial_start_frame", pd.Int32Dtype()),
    ("trial_start_pc_timestamp", pd.Int64Dtype()),
    ("trial_end_frame", pd.Int32Dtype()),
    ("trial_end_pc_timestamp", pd.Int64Dtype()),
    ("trial_pc_duration", pd.Int32Dtype()),
    ("trial_outcome", pd.Int32Dtype()),
    ("trial_start_ephys_timestamp", pd.Int64Dtype()),
    ("trial_end_ephys_timestamp", pd.Int64Dtype()),
    
    # slowly chanfing as animals runs through the zones    
    ("zone", pd.StringDtype()),
    ("frame_state", pd.Int16Dtype()),

    # chaning every 1cm
    ("from_z_position_bin", pd.Int16Dtype()),  # Nullable integer
    ("to_z_position_bin", pd.Int16Dtype()),    # Nullable integer
    
    # changing every frame if animal not 100% still
    ("frame_z_position", pd.Float32Dtype()),
    # changing every frame
    ("frame_id", pd.Int64Dtype()),
    ("frame_pc_timestamp", pd.Int64Dtype()),
    ("frame_ephys_timestamp", pd.Int64Dtype()), # Nullable integer
    ("ballvelocity_first_package", pd.Int64Dtype()),
    ("ballvelocity_last_package", pd.Int64Dtype()),
    ("frame_blinker", pd.BooleanDtype()),      # Nullable boolean

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
    
    ("L_count", pd.Int16Dtype()),
    ("R_count", pd.Int16Dtype()),
    ("S_count", pd.Int16Dtype()),
    ("V_count", pd.Int16Dtype()),
    
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
    
    ("L_count", pd.Int16Dtype()),
    ("R_count", pd.Int16Dtype()),
    ("S_count", pd.Int16Dtype()),
    ("V_count", pd.Int16Dtype()),
    
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

MULTI_UNITS_TABLE = OrderedDict([
    ("spike_time", pd.Int64Dtype()),
    ("cluster_id", pd.Int32Dtype()),
    ("spike_site", pd.Int32Dtype()),
    ("spike_color", pd.StringDtype()),
])