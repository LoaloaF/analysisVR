import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import decimate
import os
from datetime import datetime
import textwrap
from utils_plot import *

class LinearTrackPlot:
    def __init__(self, data, event_data, variable_data, metadata, animal_id, data_group):
        self.data = data
        self.event_data = event_data
        self.variable_data = variable_data
        self.metadata = metadata
        self.animal_id = animal_id
        self.data_group = data_group
        self.sessions = data["session_name"].unique().tolist()
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
        if self.data_group == "Late":
            # the start position of the track (-160 in the unity coordinate, so we should add 160 later)
            start_pos = 160 
            step = (start_pos + 260) /res
            bins = np.arange(-start_pos, start_pos + 100, step)

            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"] + start_pos)

        elif self.data_group == "16pillars":
            start_pos = 160 
            step = (start_pos + 260) /res
            bins = np.arange(-start_pos, start_pos + 100, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar9_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar12_y"] + start_pos)
        else:
            # the start position of the track (-230 in the unity coordinate, so we should add 230 later)
            start_pos = 230
            step = 2 * start_pos / res
            bins = np.arange(-start_pos, start_pos, step)
            # get the posiiton of each zone by the pillar y position
            # TODO: accommodate the newest 16-pillar adaptation
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar5_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar6_y"] + start_pos)

        reward1_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"] + start_pos)
        reward2_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"] + start_pos)

        # 2 plots for 2 cues
        for plot_id, each_session in enumerate(self.sessions):
            main_ax = fig.add_subplot(gs[plot_id, 0])

            for cue_idx in range(2):
                # get data and trial information
                data = self.data[self.data["cue_id"] == cue_idx+1]
                data.reset_index(drop=True, inplace=True)
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

                # get the mean and std of the velocity
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
            main_ax.fill_betweenx(y=[0, ymax], x1=cue_pos-20, x2=cue_pos+20, color='k', alpha=0.1, label='Cue Zone')
            main_ax.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue Zone')
            main_ax.fill_betweenx(y=[0, ymax], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.2, label='Reward 1 Zone')
            main_ax.fill_betweenx(y=[0, ymax], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.2, label='Reward 2 Zone')

            main_ax.legend(loc='upper left')
        
        fig.savefig(f"Velocity_Animal{self.animal_id}_{self.data_group}.png", dpi=300)


    def make_trialwise_velocity_heatmap(self):

        fig = plt.figure(figsize=(24, 4 * len(self.sessions)))
        gs = fig.add_gridspec(1, 2)
        
        
        res = 100
        if self.data_group == "Late":
            # the start position of the track (-160 in the unity coordinate, so we should add 160 later)
            start_pos = 160
            dist = 420
            step = (2 * start_pos + 100) /res
            bins = np.arange(-start_pos, start_pos + 100, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"] + start_pos)/step
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)/step
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"] + start_pos)/step

        elif self.data_group == "16pillars":
            start_pos = 160 
            dist = 420
            step = (2 * start_pos + 100) /res
            bins = np.arange(-start_pos, start_pos + 100, step)
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar9_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar12_y"] + start_pos)
        else:
            # the start position of the track (-230 in the unity coordinate, so we should add 230 later)
            start_pos = 230
            dist = 460
            step = 2 * start_pos / res
            bins = np.arange(-start_pos, start_pos, step)
            # get the position of each zone by the pillar y position
            # TODO: accommodate the newest 16-pillar adaptation
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar5_y"] + start_pos)/step
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"] + start_pos)/step
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar6_y"] + start_pos)/step

        reward1_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar3_y"] + start_pos)/step
        reward2_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar4_y"] + start_pos)/step

        # plot for 2 cues
        for cue_idx in range(2):
            main_ax = fig.add_subplot(gs[0, cue_idx])
            # get thy data and trial information
            data = self.data[self.data["cue_id"] == cue_idx+1]
            data.reset_index(drop=True, inplace=True)
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

                if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
                    # print the lines horizontally to separate the sessions
                    main_ax.axhline(y=row_idx, color='k', linestyle='--')
                    # print the notes of the session at the end position of the session (only print in the left plot)
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

            # print the last session notes
            if cue_idx == 0:
                wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=48))
                main_ax.text(-32, row_idx, wrapped_text, fontsize=8, ha='left')
            
            im = main_ax.imshow(velocities, aspect='auto', cmap='coolwarm', vmin=0, vmax=100)

            x_ticks = np.linspace(0, velocities.shape[1] - 1, num=5)  # Adjust the number of ticks as needed
            # Map these x-ticks to labels ranging from 0 to 420
            x_tick_labels = np.linspace(0, dist, num=5)

            # Set the x-ticks and x-tick labels
            main_ax.set_xticks(x_ticks)
            main_ax.set_xticklabels([f'{int(label)}' for label in x_tick_labels])

            # Only show the colorbar in the right plot
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
        
        fig.savefig(f"Velocity_Heatmap_Animal{self.animal_id}_{self.data_group}.png", dpi=300)


    def make_trialwise_lick_plot(self):

        fig = plt.figure(figsize=(24, 4 * len(self.sessions)))
        plt.gca().axis('off')

        # 7 columns for the 7 plots, referring to 3 plots (lick, wait_time, outcome) for each cue and 
        # 1 in the middle for the white space
        gs = fig.add_gridspec(1, 7, width_ratios=[4, 0.5, 0.5, 0.5, 4, 0.5, 0.5])

        if self.data_group == "Late":
            start_pos = -160
            end_pos = 260
            plt.xlim(-start_pos, end_pos)
            text_pos = -400 # Where to put the session notes
            # get the position of each zone by the pillar y position
            cue_enter_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar7_y"]
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"]
            cue_exit_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar8_y"]
        elif self.data_group == "16pillars":
            start_pos = -160
            end_pos = 260
            plt.xlim(-start_pos, end_pos)
            text_pos = -400 # Where to put the session notes
            cue_enter_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar9_y"] + start_pos)
            cue_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"] + start_pos)
            cue_exit_pos = (self.metadata[0]["size"]/2 - self.metadata[0]["pillar12_y"] + start_pos)
        else:
            start_pos = -230
            end_pos = 230
            plt.xlim(start_pos, end_pos)
            text_pos = -480 # Where to put the session notes
            # get the position of each zone by the pillar y position
            # TODO: accommodate the newest 16-pillar adaptation
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
            
            # get the data and trial information
            data = self.data[self.data["cue_id"] == cue_idx+1]
            data.reset_index(drop=True, inplace=True)
            lick_data = self.event_data[self.event_data["cue_id"] == cue_idx+1]
            lick_data.reset_index(drop=True, inplace=True)

            trials = data["trial_id"].unique()
            trials_variable = self.variable_data[self.variable_data["trial_id"].isin(trials)]

            session_end_trial_list = []
            sessions = data["session_name"].unique()
            for each_session in sessions:
                session_data = data[data["session_name"] == each_session]
                session_end_trial_list.append(session_data["trial_id"].max())

            # plot the zone indicators in the background at first
            lick_plot_ax.axvline(x=cue_enter_pos, color='k', linestyle='--', label='Enter Cue')
            if cue_idx == 0:
                lick_plot_ax.axvline(x=cue_pos-20, color='pink', linestyle='--')
                lick_plot_ax.axvline(x=cue_pos+20, color='pink', linestyle='--')
            elif cue_idx == 1:
                lick_plot_ax.axvline(x=cue_pos-20, color='lightblue', linestyle='--')
                lick_plot_ax.axvline(x=cue_pos+20, color='lightblue', linestyle='--')
            lick_plot_ax.axvline(x=cue_exit_pos, color='k', linestyle='--', label='Exit Cue')

            lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=reward1_pos-20, x2=reward1_pos+20, color='pink', alpha=0.6, label='Reward 1')
            lick_plot_ax.fill_betweenx(y=[0, len(trials)], x1=reward2_pos-20, x2=reward2_pos+20, color='lightblue', alpha=0.6, label='Reward 2')

            # plot the licks then
            row_idx = 0
            note_idx = 0
            for trial_id in trials:
                # plot the lines horizontally to separate the sessions
                if trial_id in session_end_trial_list and trial_id != session_end_trial_list[-1]:
                    lick_plot_ax.axhline(y=row_idx, color='k', linestyle='--')
                    # plot the notes of the session at the end position of the session (only plot in the left plot)
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
            x_ticks = np.linspace(start_pos, end_pos, num=5)  # Adjust the number of ticks as needed

            # Map these x-ticks to labels ranging from 0 to 460
            x_tick_labels = np.linspace(0, end_pos-start_pos, num=5)

            # Set the x-ticks and x-tick labels
            lick_plot_ax.set_xticks(x_ticks)
            lick_plot_ax.set_xticklabels([f'{int(label)}' for label in x_tick_labels])

            # plot the notes of the last session
            if cue_idx == 0:
                wrapped_text = "\n".join(textwrap.wrap(self.metadata[note_idx]["session_notes"], width=32))
                lick_plot_ax.text(text_pos, row_idx, wrapped_text, fontsize=10, ha='left')
            
            lick_plot_ax.legend(loc='upper left')

            # wait_time plot
            stay_times = trials_variable["stay_time"].values
            trial_ids = np.arange(len(stay_times))
            WT_plot_ax.plot(stay_times, trial_ids, color="k")
            WT_plot_ax.set_xlabel("ST (s)")
            WT_plot_ax.set_ylim(len(trials), 0)
            WT_plot_ax.set_xlim(0, 2)

            # outcome plot
            colors = ['#cd1414', '#40ca72', '#2cde6eff', '#19ed6bff', '#00ff4fff', '#62ff00ff'] 
            custom_cmap = ListedColormap(colors)
            trial_outcomes = trials_variable["trial_outcome"].values
            trial_outcomes = trial_outcomes.reshape([-1, 1])
            outcome_plot_ax.imshow(trial_outcomes, aspect='auto', cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=5)
            outcome_plot_ax.set_axis_off()
            
        fig.savefig(f"Lick_Animal{self.animal_id}_{self.data_group}.png", dpi=300)


    def make_sessionwise_timeratio_plot(self):

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 1)

        # get the position of each zone by the pillar y position
        # TODO: accommodate the newest 16-pillar adaptation
        if self.data_group == "Late":
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar2_y"]
            track_length = 430
        else:
            cue_pos = self.metadata[0]["size"]/2 - self.metadata[0]["pillar1_y"]
            track_length = 466

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
                # get the data and trial information
                data_cue = self.data[self.data["cue_id"] == cue_idx+1]
                data_cue.reset_index(drop=True, inplace=True)

                data_session = data_cue[data_cue["session_name"] == each_session]
                trials = data_session["trial_id"].unique()

                other_ratio_session = []
                cue_ratio_session = []
                reward1_ratio_session = []
                reward2_ratio_session = []
                
                for trial_id in trials:
                    data_trial = data_session[data_session["trial_id"] == trial_id]
                    # get the total time of the trial
                    total_time = data_trial["frame_pc_timestamp"].max() - data_trial["frame_pc_timestamp"].min()
                    # get the data of each zone
                    data_cue = data_trial[(data_trial["frame_z_position"] > cue_pos-20) & (data_trial["frame_z_position"] < cue_pos+20)]
                    data_reward1 = data_trial[(data_trial["frame_z_position"] > reward1_pos-20) & (data_trial["frame_z_position"] < reward1_pos+20)]
                    data_reward2 = data_trial[(data_trial["frame_z_position"] > reward2_pos-20) & (data_trial["frame_z_position"] < reward2_pos+20)]
                    # get the time ratio of each zone
                    cue_time_ratio = (data_cue["frame_pc_timestamp"].max() - data_cue["frame_pc_timestamp"].min())/total_time
                    reward1_time_ratio = (data_reward1["frame_pc_timestamp"].max() - data_reward1["frame_pc_timestamp"].min())/total_time
                    reward2_time_ratio = (data_reward2["frame_pc_timestamp"].max() - data_reward2["frame_pc_timestamp"].min())/total_time
                    other_time_ratio = 1 - cue_time_ratio - reward1_time_ratio - reward2_time_ratio
                    # get the normalized time ratio (normalized by the zone length )of each zone
                    cue_time_ratio_normalized = cue_time_ratio/40
                    reward1_time_ratio_normalized = reward1_time_ratio/40
                    reward2_time_ratio_normalized = reward2_time_ratio/40
                    other_time_ratio_normalized = other_time_ratio/(track_length-120)
                    sum_time_ratio_normalized = cue_time_ratio_normalized + reward1_time_ratio_normalized + reward2_time_ratio_normalized + other_time_ratio_normalized
                    
                    cue_ratio_session.append(cue_time_ratio_normalized / sum_time_ratio_normalized)
                    reward1_ratio_session.append(reward1_time_ratio_normalized / sum_time_ratio_normalized)
                    reward2_ratio_session.append(reward2_time_ratio / sum_time_ratio_normalized)
                    other_ratio_session.append(other_time_ratio_normalized / sum_time_ratio_normalized)
                
                # clean the nan values and calculate the mean and std of the time ratio
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

            main_ax.plot(cue_ratio_mean, label="Cue", color="blue", linewidth=2)
            main_ax.plot(reward1_ratio_mean, label="Reward1", color="red", linewidth=2)
            main_ax.plot(reward2_ratio_mean, label="Reward2", color="green", linewidth=2)
            main_ax.plot(other_ratio_mean, label="Other", color='black', linewidth=2)

            # plot the std as error bar or filled area

            # main_ax.errorbar(range(len(cue_ratio_mean)), cue_ratio_mean, yerr=cue_ratio_std, color="blue", linewidth=2, capsize=3)
            # main_ax.errorbar(range(len(reward1_ratio_mean)), reward1_ratio_mean, yerr=reward1_ratio_std, color="red", linewidth=2, capsize=3)
            # main_ax.errorbar(range(len(reward2_ratio_mean)), reward2_ratio_mean, yerr=reward2_ratio_std, color="green", linewidth=2, capsize=3)
            # main_ax.errorbar(range(len(other_ratio_mean)), other_ratio_mean, yerr=other_ratio_std, color='black', linewidth=2, capsize=3)

            # main_ax.fill_between(range(len(cue_ratio_mean)), cue_ratio_mean - cue_ratio_std, cue_ratio_mean + cue_ratio_std, color="gray", alpha=0.3)
            # main_ax.fill_between(range(len(reward1_ratio_mean)), reward1_ratio_mean - reward1_ratio_std, reward1_ratio_mean + reward1_ratio_std, color="pink", alpha=0.3)
            # main_ax.fill_between(range(len(reward2_ratio_mean)), reward2_ratio_mean - reward2_ratio_std, reward2_ratio_mean + reward2_ratio_std, color="lightblue", alpha=0.3)
            # main_ax.fill_between(range(len(other_ratio_mean)), other_ratio_mean - other_ratio_std, other_ratio_mean + other_ratio_std, color='black', alpha=0.3)

            main_ax.set_xlabel("Session")
            main_ax.set_ylabel("Time Ratio")
            main_ax.set_ylim(0, 1)
            main_ax.set_title(f"Time Ratio Cue {cue_idx+1}")
            main_ax.legend(loc='upper left')

        fig.savefig(f"Time_Ratio_Animal{self.animal_id}_{self.data_group}.png", dpi=300)


    def make_reward_time_plot(self):
        
        reward_time_sessions = []
        stay_time_sessions = []

        for session_idx, each_session in enumerate(self.sessions):
            # get the data and trial information
            data = self.data

            data_session = data[data["session_name"] == each_session]

            reward1_pos = self.metadata[session_idx]["size"]/2 - self.metadata[session_idx]["pillar3_y"]
            reward2_pos = self.metadata[session_idx]["size"]/2 - self.metadata[session_idx]["pillar4_y"]


            trials = data_session["trial_id"].unique()

            reward_time_trials = []
            
            for trial_id in trials:
                data_trial = data_session[data_session["trial_id"] == trial_id]
                # get the total time of the trial
                if data_trial["cue_id"].values[0] == 1:
                # get the data of each zone
                    data_reward = data_trial[(data_trial["frame_z_position"] > reward1_pos-20) & (data_trial["frame_z_position"] < reward1_pos+20)]
                else:
                    data_reward = data_trial[(data_trial["frame_z_position"] > reward2_pos-20) & (data_trial["frame_z_position"] < reward2_pos+20)]
                # get the time ratio of each zone
                reward_time = data_reward["frame_pc_timestamp"].max() - data_reward["frame_pc_timestamp"].min()
                reward_time_trials.append(reward_time/1e6)
            
            reward_time_sessions.append(reward_time_trials)

            trials_variable = self.variable_data[self.variable_data["trial_id"].isin(trials)]
            stay_times = trials_variable["stay_time"].values
            stay_time_sessions.append(stay_times)


        fig, ax = plt.subplots(figsize=(15, 5))

        # Flatten the reward_time_sessions and stay_time_sessions lists
        reward_times_flat = [time for session in reward_time_sessions for time in session]
        stay_times_flat = [time for session in stay_time_sessions for time in session]

        # Create a list of trial indices with gaps between sessions
        trial_indices = []
        gap = 40  # Define a gap between sessions
        current_index = 0

        for session in reward_time_sessions:
            trial_indices.extend(list(range(current_index, current_index + len(session))))
            current_index += len(session) + gap

        # Determine the colors for the scatter points
        colors = ['red' if reward_time < stay_time else '#66BB66' 
                for reward_time, stay_time in zip(reward_times_flat, stay_times_flat)]

        # Plot the reward times as scatter plot with conditional colors
        ax.scatter(trial_indices, reward_times_flat, label='Reward Time', color=colors, s=3)

        # Plot the stay times as a line plot
        ax.plot(trial_indices, stay_times_flat, label='Stay Time', color="k", linestyle='--', linewidth=2)

        # Set x-axis labels to show session_id for a group of trials
        session_labels = []
        for session_idx, session in enumerate(reward_time_sessions):
            session_labels.extend([f'{session_idx + 1}'] * len(session))

        # Set x-ticks at the start of each session
        session_ticks = [sum(len(session) for session in reward_time_sessions[:i]) + i * gap for i in range(len(reward_time_sessions))]
        ax.set_xticks(session_ticks)
        ax.set_xticklabels([f'{i + 1}' for i in range(len(reward_time_sessions))])

        ax.set_ylim(0, 5)
        ax.set_xlabel('Sessions')
        ax.set_ylabel('Time')
        ax.set_title('Reward Time and Stay Time Across Trials')
        ax.legend()

        fig.savefig(f"Reward_Time_Animal{self.animal_id}_{self.data_group}.png", dpi=300)




def generate_all_figures(animal_id, data_group):
    parent_folder = f"/mnt/NTnas/nas_vrdata/RUN_rYL00{animal_id}/rYL00{animal_id}_P0800/"

    late_start_time = datetime.strptime("2024-07-30", '%Y-%m-%d')

    data_folders = []
    for data_folder in os.listdir(parent_folder):
        if data_folder[:4] != "2024":
            continue

        data_time = datetime.strptime(data_folder[:10], '%Y-%m-%d')

        if data_group == "Early" and data_time < late_start_time:
            data_folders.append(data_folder)
        elif data_group == "Late" and data_time >= late_start_time:
            data_folders.append(data_folder)
        elif data_group == "All":
            data_folders.append(data_folder)
    
    # sort the data folders by date
    data_folders.sort()

    # get all data across sessions
    data, event_data, variable_data, metadata = get_all_data(parent_folder, data_folders)

    data, event_data = add_cue_2data(data, event_data, variable_data)

    # generate the plot object
    plot = LinearTrackPlot(data, event_data, variable_data, metadata, animal_id, data_group)

    # 4 types of plots we have now
    if animal_id == 1 or animal_id == 2 or animal_id == 3:
        if data_group != "All":
            plot.make_trialwise_velocity_plot()
            plot.make_trialwise_velocity_heatmap()
            plot.make_trialwise_lick_plot()
            plot.make_sessionwise_timeratio_plot()
        else:
            plot.make_reward_time_plot()
    else:
        # plot.make_trialwise_velocity_plot()
        plot.make_trialwise_velocity_heatmap()
        # plot.make_trialwise_lick_plot()
        # plot.make_reward_time_plot()


def main():
    # specify the animal ids and the date to generate the figures
    animal_ids = [8]
    # for animal 1,2,3: Late for after 2024-07-30, Early for before 2024-07-30
    # for animal 8: use All for all data
    data_groups = ["All"] 

    for animal_id in animal_ids:
        for data_group in data_groups:
            # generate all types of figures for this setting
            generate_all_figures(animal_id, data_group)


if __name__ == "__main__":
    main()

