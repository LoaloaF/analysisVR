import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import os
from datetime import datetime
import textwrap
import json

class LinearTrackPlot:
    def __init__(self, data, event_data, variable_data, metadata, animal_id, check_after_data):
        self.data = data
        self.event_data = event_data
        self.variable_data = variable_data
        self.metadata = metadata
        self.animal_id = animal_id
        self.data_date = "Late" if check_after_data else "Early"
        self.sessions = data[0]["session_name"].unique().tolist()
        self.sessions.sort()

    def make_trialwise_position_overtime_plot(self):
        fig = plt.figure(figsize=(30, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.5, wspace=0.2)

        for i in range(2):
            main_ax = fig.add_subplot(gs[i, 0])
            data = self.data[i]
            
            for trial_id in data["trial_id"].unique():
                if trial_id == -1:
                    continue
                trial = data[data["trial_id"] == trial_id]
                t = trial['frame_pc_timestamp'] /1e6
                # t = trial['frame_id']
                x = trial['frame_z_position']
                main_ax.plot(t, x)
            main_ax.xlabel("Time from trial start (s)")
            main_ax.ylabel("Position (a.u.)")

        fig.savefig(f"Position_Animal{self.animal_id}_{self.early_data}", dpi=300)
        fig.close()

    def make_trialwise_velocity_plot(self):

        fig = plt.figure(figsize=(15, 5 * len(self.sessions)))
        gs = fig.add_gridspec(len(self.sessions), 1)
            
        res = 100
        if self.data_date == "Late":
            start_pos = 160
            step = (2 * start_pos + 100) /res
            bins = np.arange(-start_pos, start_pos + 100, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"] + start_pos)

        else:
            start_pos = 230
            step = 2 * start_pos / res
            bins = np.arange(-start_pos, start_pos, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar5_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar6_y"] + start_pos)

        reward1_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"] + start_pos)
        reward2_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"] + start_pos)


        # plot for 2 cues
        for plot_id, each_session in enumerate(self.sessions):
            main_ax = fig.add_subplot(gs[plot_id, 0])

            for cue_idx in range(2):
                
                data = self.data[cue_idx]
                data_session = data[data["session_name"] == each_session]
                trials = data_session["trial_id"].unique()

                velocities = np.zeros((len(trials), res))
                
                row_idx = 0
                for trial_id in trials:
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

                mean_velocities = velocities.mean(axis=0)
                std_velocities = velocities.std(axis=0)

                x = np.arange(0, mean_velocities.shape[0]*step, step)
                
                if cue_idx == 0:
                    main_ax.plot(x, mean_velocities, color="purple", label='Velocity Cue 1')
                    main_ax.fill_between(x, mean_velocities - std_velocities, mean_velocities + std_velocities, color='purple', alpha=0.2)
                elif cue_idx == 1:
                    main_ax.plot(x, mean_velocities, color="green", label='Velocity Cue 2')
                    main_ax.fill_between(x, mean_velocities - std_velocities, mean_velocities + std_velocities, color='green', alpha=0.2)

            main_ax.set_xlabel("Position (cm)")
            main_ax.set_ylabel("Velocity (cm/s)")
            main_ax.set_xlim(0, mean_velocities.shape[0]*step)
            main_ax.set_ylim(0, 100)

            wrapped_text = "\n".join(textwrap.wrap(self.metadata[plot_id]["session_notes"], width=24))
            main_ax.text(-60, 0, wrapped_text, fontsize=8, ha='left')

            ymin, ymax = main_ax.get_ylim()

            main_ax.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue Zone')
            # plt.axvline(x=cue_pos, color='b', linestyle='--', label='Cue')
            main_ax.fill_betweenx(y=[0, ymax], x1=cue_pos-20, x2=cue_pos+20, color='k', alpha=0.1, label='Cue Zone')
            main_ax.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue Zone')

            main_ax.fill_betweenx(y=[0, ymax], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.2, label='Reward 1 Zone')
            main_ax.fill_betweenx(y=[0, ymax], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.2, label='Reward 2 Zone')

            main_ax.legend(loc='upper left')
        
        fig.savefig(f"Velocity_Animal{self.animal_id}_{self.data_date}", dpi=300)

    def make_trialwise_velocity_heatmap(self):

        fig = plt.figure(figsize=(24, 4 * len(self.sessions)))
        gs = fig.add_gridspec(1, 2)
            
        res = 100
        if self.data_date == "Late":
            start_pos = 160
            step = (2 * start_pos + 100) /res
            bins = np.arange(-start_pos, start_pos + 100, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"] + start_pos)/step
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)/step
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"] + start_pos)/step

        else:
            start_pos = 230
            step = 2 * start_pos / res
            bins = np.arange(-start_pos, start_pos, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar5_y"] + start_pos)/step
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"] + start_pos)/step
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar6_y"] + start_pos)/step

        reward1_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"] + start_pos)/step
        reward2_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"] + start_pos)/step


        # plot for 2 cues
        for cue_idx in range(2):
            main_ax = fig.add_subplot(gs[0, cue_idx])
            data = self.data[cue_idx]

            trials = data["trial_id"].unique()
            session_end_trial_list = []
            sessions = data["session_name"].unique()
            for each_session in sessions:
                session_data = data[data["session_name"] == each_session]
                session_end_trial_list.append(session_data["trial_id"].max())


            velocities = np.zeros((len(trials), res))
    
            row_idx = 0
            note_idx = 0
            for trial_id in trials:
                print("Processing trial ", trial_id)
                if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
                    main_ax.axhline(y=row_idx, color='k', linestyle='--')

                    if cue_idx == 0:
                        wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=48))
                        main_ax.text(-32, row_idx, wrapped_text, fontsize=8, ha='left')
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

            if cue_idx == 0:
                wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=48))
                main_ax.text(-32, row_idx, wrapped_text, fontsize=8, ha='left')
            

            im = main_ax.imshow(velocities, aspect='auto', cmap='coolwarm', vmin=0, vmax=100)

            if cue_idx == 1:
                plt.colorbar(im, ax=main_ax)
            main_ax.set_xlabel("Position (a.u.)")
            main_ax.set_xlim(0, res)
            main_ax.set_ylim(len(trials), 0)

            main_ax.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
            main_ax.axvline(x=cue_pos, color='k', linestyle='--', label='Cue')
            main_ax.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')
            main_ax.axvline(x=reward1_pos, color='purple', linestyle='--', label='Reward 1')
            main_ax.axvline(x=reward2_pos, color='blue', linestyle='--', label='Reward 2')

            main_ax.set_title(f"Velocity Heatmap Cue {cue_idx+1}")
            main_ax.legend(loc='upper left')
        
        fig.savefig(f"Velocity_Heatmap_Animal{self.animal_id}_{self.data_date}", dpi=300)

    def make_trialwise_lick_plot(self):

        fig = plt.figure(figsize=(24, 4 * len(self.sessions)))
        gs = fig.add_gridspec(1, 7, width_ratios=[4, 0.5, 0.5, 0.5, 4, 0.5, 0.5])

        if self.data_date == "Late":
            plt.xlim(-160, 260)
            text_pos = -400
            cue_enter_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"]
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"]
            cue_exit_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"]
        else:
            plt.xlim(-230, 230)
            text_pos = -480
            cue_enter_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar5_y"]
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"]
            cue_exit_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar6_y"]

        reward1_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"]
        reward2_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"]

        inter_white_space = fig.add_subplot(gs[0, 3])
        inter_white_space.set_axis_off()


        for cue_idx in range(2):
            if cue_idx == 0:
                dot_color = 'purple'
                lick_plot_ax = fig.add_subplot(gs[0, 0])
                WT_plot_ax = fig.add_subplot(gs[0, 1])
                outcome_plot_ax = fig.add_subplot(gs[0, 2])
            elif cue_idx == 1:
                dot_color = 'green'
                lick_plot_ax = fig.add_subplot(gs[0, 4])
                WT_plot_ax = fig.add_subplot(gs[0, 5])
                outcome_plot_ax = fig.add_subplot(gs[0, 6])

            data = self.data[cue_idx]
            lick_data = self.event_data[cue_idx]
            trials = data["trial_id"].unique()
            trials_variable = self.variable_data[self.variable_data["trial_id"].isin(trials)]

            session_end_trial_list = []
            sessions = data["session_name"].unique()
            for each_session in sessions:
                session_data = data[data["session_name"] == each_session]
                session_end_trial_list.append(session_data["trial_id"].max())

            row_idx = 0
            note_idx = 0
            for trial_id in trials:
                if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
                    lick_plot_ax.axhline(y=row_idx, color='k', linestyle='--')

                    if cue_idx == 0:
                        wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=32))
                        lick_plot_ax.text(text_pos, row_idx, wrapped_text, fontsize=10, ha='left')
                        note_idx += 1

                data_trial = data[data["trial_id"] == trial_id]
                trial_licks = lick_data[lick_data["trial_id"] == trial_id].event_pc_timestamp.values
                frame_timestamps = data_trial.frame_pc_timestamp.values
                frame_z_positions = data_trial.frame_z_position.values

                abs_diff = np.abs(frame_timestamps[:, np.newaxis] - trial_licks)

                closest_unity_indices = np.argmin(abs_diff, axis=0)
                x_positions = frame_z_positions[closest_unity_indices]
                lick_plot_ax.scatter(x_positions, np.full_like(x_positions, row_idx), s=4, color=dot_color, alpha=0.6)
                row_idx += 1

            lick_plot_ax.set_xlabel("Position (a.u.)")
            lick_plot_ax.set_ylim(len(trials), 0)

            if cue_idx == 0:
                wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=32))
                lick_plot_ax.text(text_pos, row_idx, wrapped_text, fontsize=10, ha='left')
            
            lick_plot_ax.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
            
            # plt.axvline(x=cue_pos, color='b', linestyle='--', label='Cue')
            lick_plot_ax.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')

            lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.6, label='Reward 1')
            lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.6, label='Reward 2')
            
            if cue_idx == 0:
                lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=cue_pos-20, x2=cue_pos+20, color='pink', alpha=0.6, label='Cue')
            elif cue_idx == 1:
                lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=cue_pos-20, x2=cue_pos+20, color='lightblue', alpha=0.6, label='Cue')

            lick_plot_ax.legend(loc='upper left')
            lick_plot_ax.set_axis_off()

            # WT plot
            stay_times = trials_variable["stay_time"].values
            trial_ids = np.arange(len(stay_times))
            WT_plot_ax.plot(stay_times, trial_ids, color="k")
            WT_plot_ax.set_xlabel("ST (s)")
            WT_plot_ax.set_ylim(len(trials), 0)
            WT_plot_ax.set_axis_off()

            
            # outcome plot
            trial_outcomes = trials_variable["trial_outcome"].values
            trial_outcomes = trial_outcomes.reshape([-1, 1])
            outcome_plot_ax.imshow(trial_outcomes, aspect='auto', cmap='afmhot', vmin=0, vmax=5)
            outcome_plot_ax.set_axis_off()
            


        fig.savefig(f"Lick_Animal{self.animal_id}_{self.data_date}", dpi=300)

    def make_sessionwise_timeratio_plot(self):

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 1)

        if self.data_date == "Late":
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"]
        else:
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"]

        reward1_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"]
        reward2_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"]

        for cue_idx in range(2):
            main_ax = fig.add_subplot(gs[cue_idx, 0])

            other_ratio_mean = []
            other_ratio_std = []
            cue_ratio_mean = []
            cue_ratio_std = []
            reward1_ratio_mean = []
            reward1_ratio_std = []
            reward2_ratio_mean = []
            reward2_ratio_std = []

            for each_session in self.sessions:
                
                data_cue = self.data[cue_idx]

                data_session = data_cue[data_cue["session_name"] == each_session]
                trials = data_session["trial_id"].unique()

                other_ratio_session = []
                cue_ratio_session = []
                reward1_ratio_session = []
                reward2_ratio_session = []
                
                for trial_id in trials:
                    data_trial = data_session[data_session["trial_id"] == trial_id]
                    total_time = data_trial["frame_pc_timestamp"].max() - data_trial["frame_pc_timestamp"].min()

                    data_cue = data_trial[(data_trial["frame_z_position"] > cue_pos-20) & (data_trial["frame_z_position"] < cue_pos+20)]
                    data_reward1 = data_trial[(data_trial["frame_z_position"] > reward1_pos-20) & (data_trial["frame_z_position"] < reward1_pos+20)]
                    data_reward2 = data_trial[(data_trial["frame_z_position"] > reward2_pos-20) & (data_trial["frame_z_position"] < reward2_pos+20)]


                    cue_time = data_cue["frame_pc_timestamp"].max() - data_cue["frame_pc_timestamp"].min()
                    reward1_time = data_reward1["frame_pc_timestamp"].max() - data_reward1["frame_pc_timestamp"].min()
                    reward2_time = data_reward2["frame_pc_timestamp"].max() - data_reward2["frame_pc_timestamp"].min()
                    
                    cue_ratio_session.append(cue_time / total_time)
                    reward1_ratio_session.append(reward1_time / total_time)
                    reward2_ratio_session.append(reward2_time / total_time)
                    other_ratio_session.append(1 - cue_time/total_time - reward1_time/total_time - reward2_time/total_time)
                
                cleaned_other_ratio_session = [x for x in other_ratio_session if not np.isnan(x)]
                other_ratio_mean.append(np.mean(cleaned_other_ratio_session))
                other_ratio_std.append(np.std(cleaned_other_ratio_session))

                cleaned_cue_ratio_session = [x for x in cue_ratio_session if not np.isnan(x)]
                cue_ratio_mean.append(np.mean(cleaned_cue_ratio_session))
                cue_ratio_std.append(np.std(cleaned_cue_ratio_session))

                cleaned_reward1_ratio_session = [x for x in reward1_ratio_session if not np.isnan(x)]
                reward1_ratio_mean.append(np.mean(cleaned_reward1_ratio_session))
                reward1_ratio_std.append(np.std(cleaned_reward1_ratio_session))

                cleaned_reward2_ratio_session = [x for x in reward2_ratio_session if not np.isnan(x)]
                reward2_ratio_mean.append(np.mean(cleaned_reward2_ratio_session))
                reward2_ratio_std.append(np.std(cleaned_reward2_ratio_session))

            cue_ratio_mean = np.array(cue_ratio_mean)
            cue_ratio_std = np.array(cue_ratio_std)
            reward1_ratio_mean = np.array(reward1_ratio_mean)
            reward1_ratio_std = np.array(reward1_ratio_std)
            reward2_ratio_mean = np.array(reward2_ratio_mean)
            reward2_ratio_std = np.array(reward2_ratio_std)
            other_ratio_mean = np.array(other_ratio_mean)
            other_ratio_std = np.array(other_ratio_std)

            main_ax.plot(cue_ratio_mean, label="Cue", color="gray")
            main_ax.plot(reward1_ratio_mean, label="Reward1", color="pink")
            main_ax.plot(reward2_ratio_mean, label="Reward2", color="lightblue")
            main_ax.plot(other_ratio_mean, label="Other", color='black')

            main_ax.fill_between(range(len(cue_ratio_mean)), cue_ratio_mean - cue_ratio_std, cue_ratio_mean + cue_ratio_std, color="gray", alpha=0.3)
            main_ax.fill_between(range(len(reward1_ratio_mean)), reward1_ratio_mean - reward1_ratio_std, reward1_ratio_mean + reward1_ratio_std, color="pink", alpha=0.3)
            main_ax.fill_between(range(len(reward2_ratio_mean)), reward2_ratio_mean - reward2_ratio_std, reward2_ratio_mean + reward2_ratio_std, color="lightblue", alpha=0.3)
            main_ax.fill_between(range(len(other_ratio_mean)), other_ratio_mean - other_ratio_std, other_ratio_mean + other_ratio_std, color='black', alpha=0.3)

            main_ax.set_xlabel("Session")
            main_ax.set_ylabel("Time Ratio")
            main_ax.set_ylim(0, 1)
            main_ax.set_title(f"Time Ratio Cue {cue_idx+1}")
            main_ax.legend(loc='upper left')

        fig.savefig(f"Time_Ratio_Animal{self.animal_id}_{self.data_date}", dpi=300)




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
            metadata["session_notes"] = "Lick sensor broken_" + metadata["session_notes"]
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
    trial_data = pd.read_hdf(fullfname, key="unity_trial")
    variable_data["trial_outcome"] = trial_data["trial_outcome"]

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


    
# def make_trialwise_velocity_plot(data, check_after_data, metadata, cue_number):
#     trials = data["trial_id"].unique()

#     session_end_trial_list = []
#     sessions = data["session_name"].unique()
#     for each_session in sessions:
#         session_data = data[data["session_name"] == each_session]
#         session_end_trial_list.append(session_data["trial_id"].max())

#     fig = plt.figure(figsize=(15, 10))
#     gs = fig.add_gridspec(len(sessions), 3, width_ratios=[4, 0.5, 0.5], height_ratios=[1, 1, 1, 1], hspace=0.5, wspace=0.2)
    

#     plt.subplots(figsize=(9.6, 3.2* len(sessions)))
#     plt.subplots_adjust(left=0.3)

#     res = 100
#     if check_after_data:
#         start_pos = 160
#         step = (2 * start_pos + 100) /res
#         bins = np.arange(-start_pos, start_pos + 100, step)
#         cue_enter_pos = (metadata[0]["size"]/2 - metadata[0]["pillar7_y"] + start_pos)/step
#         cue_pos = (metadata[0]["size"]/2 - metadata[0]["pillar2_y"] + start_pos)/step
#         cue_exit_pos = (metadata[0]["size"]/2 - metadata[0]["pillar8_y"] + start_pos)/step

#     else:
#         start_pos = 230
#         step = 2 * start_pos / res
#         bins = np.arange(-start_pos, start_pos, step)
#         cue_enter_pos = (metadata[0]["size"]/2 - metadata[0]["pillar5_y"] + start_pos)/step
#         cue_pos = (metadata[0]["size"]/2 - metadata[0]["pillar1_y"] + start_pos)/step
#         cue_exit_pos = (metadata[0]["size"]/2 - metadata[0]["pillar6_y"] + start_pos)/step

#     reward1_pos = (metadata[0]["size"]/2 - metadata[0]["pillar3_y"] + start_pos)/step
#     reward2_pos = (metadata[0]["size"]/2 - metadata[0]["pillar4_y"] + start_pos)/step

#     velocities = np.zeros((len(trials), res))
    
#     row_idx = 0
#     note_idx = 0
#     for trial_id in trials:
#         if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
#             plt.axhline(y=row_idx, color='k', linestyle='--')
#             wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=24))
#             plt.text(-60, row_idx, wrapped_text, fontsize=8, ha='left')
#             note_idx += 1
            

#         data_trial = data[data["trial_id"] == trial_id]
#         t = data_trial['frame_pc_timestamp'].values /1e6
#         x = data_trial['frame_z_position'].values
#         xbin_indices = np.digitize(x, bins)
#         for i in range(res):
#             if x[xbin_indices==i].shape[0] < 2:
#                 continue
#             v = np.gradient(x[xbin_indices==i], t[xbin_indices==i]).mean()
#             velocities[row_idx, i] = v
#         row_idx += 1

#     wrapped_text = "\n".join(textwrap.wrap(metadata[note_idx]["session_notes"], width=24))
#     plt.text(-60, row_idx, wrapped_text, fontsize=8, ha='left')
    
#     # min_vals = velocities.min(axis=0)
#     # max_vals = velocities.max(axis=0)
#     # normalized_velocities = (velocities - min_vals) / (max_vals - min_vals)
#     mean_velocities = velocities.mean(axis=0)
#     std_velocities = velocities.std(axis=0)

#     # im = plt.imshow(velocities, aspect='auto', cmap='coolwarm', vmin=0, vmax=100)
#     x = np.arange(mean_velocities.shape[0])
#     plt.plot(x, mean_velocities, label='Mean Velocity')
#     plt.fill_between(x, mean_velocities - std_velocities, mean_velocities + std_velocities, color='b', alpha=0.2, label='Standard Deviation')


#     plt.xlabel("Position (a.u.)")
#     plt.ylabel("Trial ID")
#     # plt.colorbar(im)
#     plt.ylim(len(trials), 0)

#     plt.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
#     # plt.axvline(x=cue_pos, color='b', linestyle='--', label='Cue')
#     plt.fill_betweenx(y=[0, len(trials)], x1=cue_pos-20, x2=cue_pos+20, color='pink', alpha=0.6, label='Cue')
#     plt.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')

#     if cue_number == 1:
#         # plt.axvline(x=reward1_pos, color='r', linestyle='--', label='Reward 1')
#         # plt.axvline(x=reward2_pos, color='k', linestyle='--', label='Reward 2')
#         plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.6, label='Reward 1')

#         plt.axvline(x=reward2_pos-20, color='lightblue', linestyle='--')
#         plt.axvline(x=reward2_pos+20, color='lightblue', linestyle='--')
#         # plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.6, label='Reward 2')
#     elif cue_number == 2:
#         # plt.axvline(x=reward1_pos, color='k', linestyle='--', label='Reward 1')
#         # plt.axvline(x=reward2_pos, color='r', linestyle='--', label='Reward 2')
#         # plt.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='lightblue', alpha=0.6, label='Reward 1')
#         plt.axvline(x=reward1_pos-20, color='lightblue', linestyle='--')
#         plt.axvline(x=reward1_pos+20, color='lightblue', linestyle='--')
#         plt.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='pink', alpha=0.6, label='Reward 2')

#     plt.legend(loc='upper left')



def generate_all_figures(animal_id, check_after_data):
    parent_folder = f"/mnt/NTnas/nas_vrdata/RUN_rYL00{animal_id}/rYL00{animal_id}_P0800/"
    real_start_date = datetime.strptime("2024-07-30", '%Y-%m-%d')

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

    trial_1= variable_data[variable_data["cue"] == 1]["trial_id"]
    trial_2= variable_data[variable_data["cue"] == 2]["trial_id"]
    data_1, event_data_1 = selected_data_by_cue(data, event_data, trial_1)
    data_2, event_data_2 = selected_data_by_cue(data, event_data, trial_2)

    plot = LinearTrackPlot([data_1, data_2], [event_data_1, event_data_2], variable_data, metadata, animal_id, check_after_data)

    plot.make_trialwise_velocity_plot()
    plot.make_trialwise_velocity_heatmap()
    plot.make_trialwise_lick_plot()
    plot.make_sessionwise_timeratio_plot()

    # if cue_number != 0:
    #     trial_selected = variable_data[variable_data["cue"] == cue_number]["trial_id"]
    #     data_selected, event_data_selected = selected_data_by_cue(data, event_data, trial_selected)
    # else:
    #     data_selected = data
    #     event_data_selected = event_data

    

    # fig1 = plt.figure(figsize=(10, 5))
    # make_trialwise_position_overtime_plot(data)
    # figure_type = "Position"
    # fig1.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    # fig1.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    # fig1.close()

    # make_trialwise_velocity_plot(data_selected, check_after_data, metadata, cue_number)
    # figure_type = "Velocity"
    # plt.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    # plt.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    # plt.close()

    # make_trialwise_lick_plot(data_selected, event_data_selected, check_after_data, metadata, cue_number)
    # figure_type = "Lick"
    # plt.title(f"{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}")
    # plt.savefig(f"/home/ntgroup/Documents/figures/{figure_type}_Animal{animal_id}_Cue{cue_number}_{figure_date}.png", dpi=300)
    # plt.close()

    # print(f"Finish figures for Animal{animal_id}_Cue{cue_number}_{figure_date}")



def main():
    animal_ids = [1,2,3]
    check_after_datas = [True,False]

    for animal_id in animal_ids:
        # for cue_number in cue_numbers:
        for check_after_data in check_after_datas:
            generate_all_figures(animal_id, check_after_data)

if __name__ == "__main__":
    main()

