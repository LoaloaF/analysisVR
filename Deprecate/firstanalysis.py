import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

class TrialWiseVelocity:
    def __init__(self, figsize=(10, 5)):
        self.figsize = figsize
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        self.ax = ax
    
    def attach_data(self, data):
        self.data = data

def get_data(with_lick = False):
    # fullfname = "/Users/loaloa/local_data/2024-07-26_14-57_rYL003_P0800_LinearTrack_27min/behavior_2024-07-26_14-57_rYL003_P0800_LinearTrack_27min.hdf5"
    fullfname = "/Users/loaloa/local_data/2024-07-26_15-33_rYL002_P0800_LinearTrack_34min/behavior_2024-07-26_15-33_rYL002_P0800_LinearTrack_34min.hdf5"
    data = pd.read_hdf(fullfname, key="unity_frame")
    if  with_lick:
        eventdata = pd.read_hdf(fullfname, key="event")
        eventdata = eventdata[eventdata["event_name"] == "L"]
        
        # print(data)
        # print(eventdata)
        # exit()
        
    def zero_t(data):
        trial_start_t = data['frame_pc_timestamp'].iloc[0]
        data['frame_pc_timestamp'] -= trial_start_t
        data['frame_id'] -= data['frame_id'].iloc[0]
        if with_lick:
            eventdata.loc[eventdata.trial_id == data.trial_id.iloc[0],"event_pc_timestamp"] -= trial_start_t
        return data
        
    data = data.groupby("trial_id").apply(zero_t)
    if with_lick:
        return data, eventdata
    return data

def make_trialwise_position_overtime_plot():
    plot = TrialWiseVelocity()
    data = get_data()
    
    plot.attach_data(data)
    for trial_id in data.index.get_level_values("trial_id").unique():
        if trial_id == -1:
            continue
        trial = data.loc[trial_id]
        t = trial['frame_pc_timestamp'] /1e6
        # t = trial['frame_id']
        x = trial['frame_z_position']
        plot.ax.plot(t, x)
    plt.xlabel("Time from trial start (s)")
    plt.ylabel("Position (a.u.)")
    plt.savefig("position_overtime.png")
    plt.show()
    
def make_trialwise_volocity_plot():
    
    data = get_data()
    trials = data.index.get_level_values("trial_id").unique().values
    res = 100
    step = 233*2 /res
    bins = np.arange(-233, 233, step)
    print(bins)
    velocities = np.zeros((len(trials), res))
    
    for trial_id in trials:
        if trial_id == -1:
            continue
        trial = data.loc[trial_id]
        t = trial['frame_pc_timestamp'].values /1e6
        x = trial['frame_z_position'].values
        xbin_indices = np.digitize(x, bins)
        print(xbin_indices)
        # v = np.gradient(x, t)
        
        # print(indices.shape)
        for i in range(res):
            # print(x[xbin_indices])
            # print(t[xbin_indices])
            # print(indices)
            # print(indices == i)
            # print(v[indices == i])
            if x[xbin_indices==i].shape[0] <2:
                continue
            print(x[xbin_indices==i], t[xbin_indices==i])
            v = np.gradient(x[xbin_indices==i], t[xbin_indices==i]).mean()
            print(v)
            velocities[int(trial_id), i] = v
            # plt.scatter(trial_id, v)

        #     break
        # break
    # plt.show()
    im = plt.imshow(velocities, aspect='auto', cmap='coolwarm')
    plt.xlabel("Position (a.u.)")
    plt.ylabel("Trial ID")
    plt.colorbar(im)
    plt.savefig("velocity.png")
    plt.show()
    
def make_trialwise_lick_plot():
    data, lickdata = get_data(with_lick=True)
    plt.subplots(figsize=(13, 4))
    # print(data)
    print(lickdata)
    
    trials = data.index.get_level_values("trial_id").unique().values
    res = 100
    step = 233*2 /100
    bins = np.arange(-233, 233, step)

    licks = []    
    for trial_id in trials:
        if trial_id == -1:
            continue
        trial = data.loc[trial_id]
        print(trial.frame_pc_timestamp.values)
        trial_licks = lickdata[lickdata["trial_id"] == trial_id].event_pc_timestamp.values
        for l in trial_licks:
            print(l)
            clostest_unity_idx = np.where(np.argsort(np.abs(trial.frame_pc_timestamp.values - l)) == 0)[0][0]
            print(clostest_unity_idx)
            try:
                x = trial.iloc[clostest_unity_idx].loc["frame_z_position"]
                print(x)
                plt.scatter(x, trial_id, s = 2, color='k', alpha=.6)
            except Exception as e:
                print(e)
                continue

    plt.ylabel("Trial ID")
    plt.xlabel("Position (a.u.)")
    plt.title("Lick events")
    plt.ylim(len(trials), 0)
    
    plt.savefig("licks.png")
    plt.show()
    
    
def main():
    make_trialwise_position_overtime_plot()
    make_trialwise_volocity_plot()
    make_trialwise_lick_plot()
    
if __name__ == "__main__":
    main()