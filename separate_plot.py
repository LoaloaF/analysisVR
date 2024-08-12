import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import os
from datetime import datetime
import textwrap
import json

class TrialWiseVelocity:
    def __init__(self, figsize=(10, 5)):
        self.figsize = figsize
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        self.ax = ax
    
    def attach_data(self, data):
        self.data = data

def refractor_data(data, data_folder, last_trial_id):
    data["session_name"] =  data_folder.split("/")[-1]
    data = data.drop(data[data['trial_id'] < 0].index)
    data["trial_id"] += last_trial_id
    return data

def extract_useful_metadata(session_parameters):
    extracted_metadata = {}

    if session_parameters["notes"][0] is None:
        extracted_metadata["session_notes"] = session_parameters["start_time"][0]
    elif session_parameters["notes"][0] != "Free notes here":
        extracted_metadata["session_notes"] = session_parameters["start_time"][0] + "_" + ''.join(session_parameters["notes"][0])
    else:
        extracted_metadata["session_notes"] = session_parameters["start_time"][0]

    session_metadata = json.loads(session_parameters["metadata"][0])
    pillar_info = session_metadata["pillars"]

    for i in range(1, 11):
        extracted_metadata[f"pillar{i}_y"] = [value["y"] for key, value in pillar_info.items() if value['id'] == i][0]

    extracted_metadata["size"] = session_metadata["envX_size"]
    return extracted_metadata

def exclude_inproper_data(data_folder, data, event_data, variable_data, metadata):

    problem_session = ["2024-07-30_14-56_rYL003_P0800_LinearTrack_52min",
                       "2024-08-02_17-15_rYL003_P0800_LinearTrack_53min",
                       "2024-07-24",
                       "2024-07-31",
                       "2024-08-02"]

    for each_session in problem_session:
        if each_session in data_folder:
            event_data = event_data.iloc[0:0]
            metadata["session_notes"] = "Problematic session"
            break

    return data, event_data, variable_data, metadata


def get_data(data_folder, last_trial_id):

    for filename in os.listdir(data_folder):
        # Check if the file name starts with the specified prefix
        if filename.startswith("behavior"):
            fullfname = os.path.join(data_folder, filename)
            break
    
    data = pd.read_hdf(fullfname, key="unity_frame")
    variable_data = pd.read_hdf(fullfname, key="paradigm_variable")
    event_data = pd.read_hdf(fullfname, key="event")
    event_data = event_data[event_data["event_name"] == "L"]

    session_parameters = pd.read_hdf(fullfname, key="metadata")
    metadata = extract_useful_metadata(session_parameters)

    # eliminate some trials
    data, event_data, variable_data, metadata = exclude_inproper_data(data_folder, data, event_data, variable_data, metadata)

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

    data = []
    event_data = []
    variable_data = []
    metadata = []
    last_trial_id = 0
    for data_folder in data_folders:

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
    data = data[data["trial_id"].isin(trial_selected)]
    data.reset_index(drop=True, inplace=True)
    event_data = event_data[event_data["trial_id"].isin(trial_selected)]
    event_data.reset_index(drop=True, inplace=True)
    return data, event_data


def make_trialwise_position_overtime_plot(data):
    plot = TrialWiseVelocity()

    plot.attach_data(data)
    for trial_id in data["trial_id"].unique():
        if trial_id == -1:
            continue
        trial = data[data["trial_id"] == trial_id]
        t = trial['frame_pc_timestamp'] /1e6
        # t = trial['frame_id']
        x = trial['frame_z_position']
        plot.ax.plot(t, x)
    plt.xlabel("Time from trial start (s)")
    plt.ylabel("Position (a.u.)")

    
def make_trialwise_velocity_plot(data, check_after_data, metadata, cue_number):
    trials = data["trial_id"].unique()

    session_end_trial_list = []
    sessions = data["session_name"].unique()
    for each_session in sessions:
        session_data = data[data["session_name"] == each_session]
        session_end_trial_list.append(session_data["trial_id"].max())

    # fig = plt.figure(figsize=(15, 10))
    # gs = fig.add_gridspec(4, 3, width_ratios=[4, 0.5, 0.5], height_ratios=[1, 1, 1, 1], hspace=0.5, wspace=0.2)
    

    plt.subplots(figsize=(9.6, 3.2* len(sessions)))
    plt.subplots_adjust(left=0.3)

    res = 100
    if check_after_data:
        start_pos = 160
        step = (2 * start_pos + 100) /res
        bins = np.arange(-start_pos, start_pos + 100, step)
        cue_enter_pos = (metadata[0]["size"]/2 - metadata[0]["pillar7_y"] + start_pos)/step
        cue_pos = (metadata[0]["size"]/2 - metadata[0]["pillar2_y"] + start_pos)/step
        cue_exit_pos = (metadata[0]["size"]/2 - metadata[0]["pillar8_y"] + start_pos)/step

    else:
        start_pos = 230
        step = 2 * start_pos / res
        bins = np.arange(-start_pos, start_pos, step)
        cue_enter_pos = (metadata[0]["size"]/2 - metadata[0]["pillar5_y"] + start_pos)/step
        cue_pos = (metadata[0]["size"]/2 - metadata[0]["pillar1_y"] + start_pos)/step
        cue_exit_pos = (metadata[0]["size"]/2 - metadata[0]["pillar6_y"] + start_pos)/step

    reward1_pos = (metadata[0]["size"]/2 - metadata[0]["pillar3_y"] + start_pos)/step
    reward2_pos = (metadata[0]["size"]/2 - metadata[0]["pillar4_y"] + start_pos)/step

    velocities = np.zeros((len(trials), res))
    
    row_idx = 0
    note_idx = 0
    for trial_id in trials:
        if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
            plt.axhline(y=row_idx, color='k', linestyle='--')
            wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=24))
            plt.text(-60, row_idx, wrapped_text, fontsize=8, ha='left')
            note_idx += 1
            

        data_trial = data[data["trial_id"] == trial_id]
        t = data_trial['frame_pc_timestamp'].values /1e6
        x = data_trial['frame_z_position'].values
        xbin_indices = np.digitize(x, bins)
        for i in range(res):
            if x[xbin_indices==i].shape[0] < 2:
                continue
            v = np.gradient(x[xbin_indices==i], t[xbin_indices==i]).mean()
            velocities[row_idx, i] = v
        row_idx += 1

    wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=24))
    plt.text(-60, row_idx, wrapped_text, fontsize=8, ha='left')
    
    # min_vals = velocities.min(axis=0)
    # max_vals = velocities.max(axis=0)
    # normalized_velocities = (velocities - min_vals) / (max_vals - min_vals)
    # mean_velocities = velocities.mean(axis=0)
    # std_velocities = velocities.std(axis=0)

    # im = plt.imshow(velocities, aspect='auto', cmap='coolwarm', vmin=0, vmax=100)
    plt.xlabel("Position (a.u.)")
    plt.ylabel("Trial ID")
    # plt.colorbar(im)
    plt.ylim(len(trials), 0)

    plt.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
    # plt.axvline(x=cue_pos, color='b', linestyle='--', label='Cue')
    plt.fill_betweenx(y=[0, len(trials)], x1=cue_pos-20, x2=cue_pos+20, color='pink', alpha=0.6, label='Cue')
    plt.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')

    if cue_number == 1:
        # plt.axvline(x=reward1_pos, color='r', linestyle='--', label='Reward 1')
        # plt.axvline(x=reward2_pos, color='k', linestyle='--', label='Reward 2')
        plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.6, label='Reward 1')

        plt.axvline(x=reward2_pos-20, color='lightblue', linestyle='--')
        plt.axvline(x=reward2_pos+20, color='lightblue', linestyle='--')
        # plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.6, label='Reward 2')
    elif cue_number == 2:
        # plt.axvline(x=reward1_pos, color='k', linestyle='--', label='Reward 1')
        # plt.axvline(x=reward2_pos, color='r', linestyle='--', label='Reward 2')
        # plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='lightblue', alpha=0.6, label='Reward 1')
        plt.axvline(x=reward1_pos-20, color='lightblue', linestyle='--')
        plt.axvline(x=reward1_pos+20, color='lightblue', linestyle='--')
        plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='pink', alpha=0.6, label='Reward 2')

    plt.legend(loc='upper left')

    
def make_trialwise_lick_plot(data, lickdata, check_after_data, metadata, cue_number):

    trials = data["trial_id"].unique()
    session_end_trial_list = []
    sessions = data["session_name"].unique()
    for each_session in sessions:
        session_data = data[data["session_name"] == each_session]
        session_end_trial_list.append(session_data["trial_id"].max())

    plt.subplots(figsize=(13, 4 * len(sessions)))
    plt.subplots_adjust(left=0.25)

    if check_after_data:
        plt.xlim(-160, 260)
        text_pos = -310
        cue_enter_pos = metadata[0]["size"]/2 - metadata[0]["pillar7_y"]
        cue_pos = metadata[0]["size"]/2 - metadata[0]["pillar2_y"]
        cue_exit_pos = metadata[0]["size"]/2 - metadata[0]["pillar8_y"]
    else:
        plt.xlim(-230, 230)
        text_pos = -380
        cue_enter_pos = metadata[0]["size"]/2 - metadata[0]["pillar5_y"]
        cue_pos = metadata[0]["size"]/2 - metadata[0]["pillar1_y"]
        cue_exit_pos = metadata[0]["size"]/2 - metadata[0]["pillar6_y"]
    
    if cue_number == 1:
        dot_color = 'purple'
    elif cue_number == 2:
        dot_color = 'green'


    reward1_pos = metadata[0]["size"]/2 - metadata[0]["pillar3_y"]
    reward2_pos = metadata[0]["size"]/2 - metadata[0]["pillar4_y"]

    row_idx = 0
    note_idx = 0
    for trial_id in trials:
        if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
            plt.axhline(y=row_idx, color='k', linestyle='--')
            wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=32))
            plt.text(text_pos, row_idx, wrapped_text, fontsize=10, ha='left')
            note_idx += 1

        data_trial = data[data["trial_id"] == trial_id]
        trial_licks = lickdata[lickdata["trial_id"] == trial_id].event_pc_timestamp.values
        for l in trial_licks:
            clostest_unity_idx = np.argsort(np.abs(data_trial.frame_pc_timestamp.values - l))[0]
            try:
                x = data_trial.iloc[clostest_unity_idx].loc["frame_z_position"]
                # print(x)
                plt.scatter(x, row_idx, s = 4, color=dot_color, alpha=.6)
            except Exception as e:
                print(e)
                continue
        row_idx += 1

    plt.ylabel("Trial ID")
    plt.xlabel("Position (a.u.)")
    plt.ylim(len(trials), 0)

    wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=32))
    plt.text(text_pos, row_idx, wrapped_text, fontsize=10, ha='left')
    
    plt.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
    plt.fill_betweenx(y=[0, len(trials)], x1=cue_pos-20, x2=cue_pos+20, color='pink', alpha=0.6, label='Cue')
    # plt.axvline(x=cue_pos, color='b', linestyle='--', label='Cue')
    plt.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')


    if cue_number == 1:
        plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.6, label='Reward 1')
        plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.6, label='Reward 2')
    elif cue_number == 2:
        plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='lightblue', alpha=0.6, label='Reward 1')
        plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='pink', alpha=0.6, label='Reward 2')

    plt.legend(loc='upper left')


def generate_all_figures(animal_id, cue_number, check_after_data):
    parent_folder = f"/mnt/NTnas/nas_vrdata/RUN_rYLI00{animal_id}/rYL00{animal_id}_P0800/"
    real_start_date = datetime.strptime("2024-07-30", '%Y-%m-%d')
    figure_date = "new" if check_after_data else "old"

    data_folders = []
    for data_folder in os.listdir(parent_folder):
        data_date = datetime.strptime(data_folder[:10], '%Y-%m-%d')
        if data_date < real_start_date and check_after_data:
            continue
        elif data_date >= real_start_date and not check_after_data:
            continue
        data_folders.append(data_folder)
    
    data_folders.sort()

    data, event_data, variable_data, metadata = get_all_data(parent_folder, data_folders)

    if cue_number != 0:
        trial_selected = variable_data[variable_data["cue"] == cue_number]["trial_id"]
        data_selected, event_data_selected = selected_data_by_cue(data, event_data, trial_selected)
    else:
        data_selected = data
        event_data_selected = event_data


    # make_trialwise_position_overtime_plot(data)
    # figure_type = "Position"
    # plt.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    # plt.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    # plt.close()

    make_trialwise_velocity_plot(data_selected, check_after_data, metadata, cue_number)
    figure_type = "Velocity"
    plt.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    plt.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    plt.close()

    # make_trialwise_lick_plot(data_selected, event_data_selected, check_after_data, metadata, cue_number)
    # figure_type = "Lick"
    # plt.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    # plt.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    # plt.close()

    print(f"Finish figures for Animal{animal_id}_Cue{cue_number}_{figure_date}")



def main():
    animal_ids = [1,2,3]
    cue_numbers = [1,2] # 0 for all cues, 1 for cue 1, 2 for cue 2
    check_after_datas = [True,False]

    for animal_id in animal_ids:
        for cue_number in cue_numbers:
            for check_after_data in check_after_datas:
                generate_all_figures(animal_id, cue_number, check_after_data)

if __name__ == "__main__":
    main()