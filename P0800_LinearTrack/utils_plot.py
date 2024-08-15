import pandas as pd
from scipy.signal import decimate
import os
import json

def exclude_inproper_data(data_folder, data, event_data, variable_data, metadata):

    # session with the lick sensor broken
    problem_session = ["2024-07-30_14-56_rYL003_P0800_LinearTrack_52min",
                       "2024-08-02_17-15_rYL003_P0800_LinearTrack_53min",
                       "2024-07-24",
                       "2024-07-31",
                       "2024-08-02"]

    for each_session in problem_session:
        if each_session in data_folder:
            event_data = event_data.iloc[0:0]
            metadata["session_notes"] = "Lick sensor broken_" + metadata["session_notes"]
            break

    return data, event_data, variable_data, metadata


def refractor_data(data, data_folder, last_trial_id):
    # add session name to the data
    data["session_name"] =  data_folder.split("/")[-1]
    # remove the data with trial_id < 0
    data = data.drop(data[data['trial_id'] < 0].index)
    # add the last trial_id to the data, so all sessions are concatenated
    data["trial_id"] += last_trial_id
    return data


def extract_useful_metadata(session_parameters):
    # extract metadata which will be used later from the session_parameters
    extracted_metadata = {}

    if session_parameters["notes"][0] is None:
        extracted_metadata["session_notes"] = session_parameters["start_time"][0]
    elif session_parameters["notes"][0] != "Free notes here":
        extracted_metadata["session_notes"] = session_parameters["start_time"][0] + "_" + ''.join(session_parameters["notes"][0])
    else:
        extracted_metadata["session_notes"] = session_parameters["start_time"][0]

    session_metadata = json.loads(session_parameters["metadata"][0])
    pillar_info = session_metadata["pillars"]

    # TODO: accommodate the newest 16-pillar adaptation
    for i in range(1, 11):
        extracted_metadata[f"pillar{i}_y"] = [value["y"] for key, value in pillar_info.items() if value['id'] == i][0]

    extracted_metadata["size"] = session_metadata["envX_size"]
    return extracted_metadata


def get_data(data_folder, last_trial_id):

    for filename in os.listdir(data_folder):
        # Check if the file name starts with the "behavior" prefix
        # TODO: accommodate the new changes without "behavior" prefix
        if filename.startswith("behavior"):
            fullfname = os.path.join(data_folder, filename)
            break
    
    # get unity data
    data = pd.read_hdf(fullfname, key="unity_frame")
    # get paradigm variable data
    variable_data = pd.read_hdf(fullfname, key="paradigm_variable")
    # get trial data, add the outcome to the variable data
    trial_data = pd.read_hdf(fullfname, key="unity_trial")
    variable_data["trial_outcome"] = trial_data["trial_outcome"]
    # get event data
    event_data = pd.read_hdf(fullfname, key="event")
    event_data = event_data[event_data["event_name"] == "L"]
    # get metadata
    session_parameters = pd.read_hdf(fullfname, key="metadata")
    metadata = extract_useful_metadata(session_parameters)

    # eliminate some problematic trials
    data, event_data, variable_data, metadata = exclude_inproper_data(data_folder, data, event_data, variable_data, metadata)
    # refractor the data to make following analysis easier
    data = refractor_data(data, data_folder, last_trial_id)
    event_data = refractor_data(event_data, data_folder, last_trial_id)
    variable_data = refractor_data(variable_data, data_folder, last_trial_id)

    def zero_t(data):
        trial_start_t = data['frame_pc_timestamp'].iloc[0]
        data['frame_pc_timestamp'] -= trial_start_t
        data['frame_id'] -= data['frame_id'].iloc[0]
        event_data.loc[event_data.trial_id == data.trial_id.iloc[0],"event_pc_timestamp"] -= trial_start_t
        return data
        
    data = data.groupby("trial_id").apply(zero_t)
    return data, event_data, variable_data, metadata


def get_all_data(parent_folder, data_folders):

    # get data for all sessions
    data = []
    event_data = []
    variable_data = []
    metadata = []
    last_trial_id = 0

    for data_folder in data_folders:
        # get the data for each session
        data_tem, event_data_tem, variable_data_tem, metadata_tem = get_data(os.path.join(parent_folder, data_folder), last_trial_id)
        last_trial_id = data_tem["trial_id"].max()

        data.append(data_tem)
        event_data.append(event_data_tem)
        variable_data.append(variable_data_tem)
        metadata.append(metadata_tem)
        print("Finish loading data from ", data_folder)

    data = pd.concat(data, axis=0, ignore_index=True)
    event_data = pd.concat(event_data, axis=0, ignore_index=True)
    variable_data = pd.concat(variable_data, axis=0, ignore_index=True)

    return data, event_data, variable_data, metadata

def selected_data_by_cue(data, event_data, trial_selected):
    # select the data and event data by the trial id of specific cue
    data = data[data["trial_id"].isin(trial_selected)]
    data.reset_index(drop=True, inplace=True)
    event_data = event_data[event_data["trial_id"].isin(trial_selected)]
    event_data.reset_index(drop=True, inplace=True)
    return data, event_data