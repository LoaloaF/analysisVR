import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def detect_edges(signal, signal_name):

    diff_val = np.diff(signal[signal_name])
    rising_edges_idx = np.where(diff_val == 1)[0] + 1
    falling_edges_idx = np.where(diff_val == -1)[0] + 1
    
    rising_edge_times = signal['time'].iloc[rising_edges_idx].values
    falling_edge_times = signal['time'].iloc[falling_edges_idx].values

    return rising_edge_times, falling_edge_times



def get_data():
    # folder = '/home/ntgroup/Project/data/2024-08-14_14-31_dummyAnimal_P0800_LinearTrack_3min/'
    # folder = '/home/ntgroup/Project/data/2024-08-16_14-40-39_active/'
    folder = "/home/ntgroup/Project/data/2024-08-21_16-21-13_active/"
    # folder = "/mnt/NTnas/nas_vrdata/Unprocessed/2024-08-19_17-53_rYL008_P0500_MotorLearning_19min/"
    ephys_fname = "ephys_output.raw.h5"
    # behavior_fname = "behavior_2024-08-14_14-31_dummyAnimal_P0800_LinearTrack_3min.hdf5"
    # behavior_fname = "2024-08-16_14-41_dummyAnimal_P0800_LinearTrack_2min.hdf5"
    behavior_fname = "2024-08-21_16-22_dummyAnimal_P0500_MotorLearning_0min.hdf5"
    # behavior_fname = "2024-08-19_17-53_rYL008_P0500_MotorLearning_19min.hdf5"

    ephys_fullfname = os.path.join(folder, ephys_fname)
    behavioral_fullfname = os.path.join(folder, behavior_fname)

    with h5py.File(ephys_fullfname, 'r') as file:
        ephys_bits = file['bits']["0000"][:]

    # convert the value into single bits and create time series
    msg = np.array([(a[0],a[1]) for a in ephys_bits])

    ephys_data = pd.DataFrame(msg, columns=['time', 'value'])

    for column_id in range(8):
        ephys_data[f'bit{column_id}'] = (ephys_data['value'] & (1 << column_id))/2**column_id
        ephys_data[f'bit{column_id}'] = ephys_data[f'bit{column_id}'].astype(int)
    

    event_data = pd.read_hdf(behavioral_fullfname, key="event")
    frame_data = pd.read_hdf(behavioral_fullfname, key="unity_frame")
    ball_data = pd.read_hdf(behavioral_fullfname, key="ballvelocity")
    facecam_data = pd.read_hdf(behavioral_fullfname, key="facecam_packages")
    return ephys_data, event_data, frame_data, ball_data, facecam_data

def comparision_plot(ttl,pc_timestamp):
    fig, ax = plt.subplots()

    # Plot the dots
    ax.scatter(ttl, np.ones_like(ttl), color='blue', label='Rising TTL')
    ax.scatter(pc_timestamp, np.zeros_like(pc_timestamp), color='red', label='PC Timestamp')

    # Draw lines between the pairs
    for i in range(len(ttl)):
        ax.plot([ttl[i], pc_timestamp[i]], [1, 0], color='gray', linestyle='--')
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['PC Timestamp', 'Rising TTL'])
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_title('Comparison of Rising TTL and PC Timestamp')

    plt.show()


def main():

    ephys_data, event_data, frame_data, ball_data, facecam_data = get_data()

    ball_ttl = ephys_data[['time', 'bit0']]
    ball_rising_ttl, ball_falling_ttl = detect_edges(ball_ttl, "bit0")

    if (ball_ttl["bit0"].iloc[0] == 1):
        ball_rising_ttl = np.insert(ball_rising_ttl, 0, ball_ttl["time"].iloc[0])

    ball_pc_timestamp = np.array(ball_data["ballvelocity_pc_timestamp"])
    ball_portenta_timestamp = np.array(ball_data["ballvelocity_portenta_timestamp"])

    ball_rising_ttl_norm = ball_rising_ttl - ball_rising_ttl[0]
    ball_pc_timestamp_norm = ball_pc_timestamp - ball_pc_timestamp[0]
    ball_portenta_timestamp_norm = ball_portenta_timestamp - ball_portenta_timestamp[0]

    ball_rising_ttl_norm = ball_rising_ttl_norm * 50

    comparision_plot(ball_rising_ttl_norm[:1000], ball_pc_timestamp_norm[:1000])
    comparision_plot(ball_rising_ttl_norm[-1000:], ball_pc_timestamp_norm[-1002:-2])
    plt.plot(ball_pc_timestamp_norm[0:-2] - ball_rising_ttl_norm, label='Rising TTL')
    plt.show()






    frame_ttl = ephys_data[['time', 'bit2']]
    frame_rising_ttl, frame_falling_ttl = detect_edges(frame_ttl, "bit2")
    combined_ttl = np.concatenate((frame_rising_ttl, frame_falling_ttl))
    combined_ttl = np.sort(combined_ttl)
    frame_pc_timestamp = np.array(frame_data["frame_pc_timestamp"])
    frame_ttl_norm = combined_ttl - combined_ttl[0]
    frame_pc_timestamp_norm = frame_pc_timestamp - frame_pc_timestamp[0]

    facecam_ttl = ephys_data[['time', 'bit3']]
    facecam_rising_ttl, facecam_falling_ttl = detect_edges(facecam_ttl, "bit3")
    facecam_pc_timestamp = np.array(facecam_data["facecam_image_pc_timestamp"])
    facecam_rising_ttl_norm = facecam_rising_ttl - facecam_rising_ttl[0]
    facecam_pc_timestamp_norm = facecam_pc_timestamp - facecam_pc_timestamp[0]

    facecam_rising_ttl_norm = facecam_rising_ttl_norm * 50
    comparision_plot(facecam_rising_ttl_norm, facecam_pc_timestamp_norm)

    lick_ttl = ephys_data[['time', 'bit5']]
    lick_rising_ttl, lick_falling_ttl = detect_edges(lick_ttl, "bit5")
    lick_pc_timestamp = np.array(event_data[event_data["event_name"]=="L"]["event_pc_timestamp"])
    lick_rising_ttl_norm = lick_rising_ttl - lick_rising_ttl[0]
    lick_pc_timestamp_norm = lick_pc_timestamp - lick_pc_timestamp[0]

    lick_rising_ttl_norm = lick_rising_ttl_norm * 50
    comparision_plot(lick_rising_ttl_norm, lick_pc_timestamp_norm)

    reward_ttl = ephys_data[['time', 'bit6']]
    reward_rising_ttl, reward_falling_ttl = detect_edges(reward_ttl, "bit6")
    reward_pc_timestamp = np.array(event_data[event_data["event_name"]=="R"]["event_pc_timestamp"])
    reward_rising_ttl_norm = reward_rising_ttl - reward_rising_ttl[0]
    reward_pc_timestamp_norm = reward_pc_timestamp - reward_pc_timestamp[0]

    reward_rising_ttl_norm = reward_rising_ttl_norm * 50
    comparision_plot(reward_rising_ttl_norm, reward_pc_timestamp_norm)


    sound_ttl = ephys_data[['time', 'bit7']]
    sound_rising_ttl, sound_falling_ttl = detect_edges(sound_ttl, "bit7")
    sound_pc_timestamp = np.array(event_data[event_data["event_name"]=="S"]["event_pc_timestamp"])
    sound_rising_ttl_norm = sound_rising_ttl - sound_rising_ttl[0]
    sound_pc_timestamp_norm = sound_pc_timestamp - sound_pc_timestamp[0]

    sound_rising_ttl_norm = sound_rising_ttl_norm * 50
    comparision_plot(sound_rising_ttl_norm, sound_pc_timestamp_norm)


    # assert len(sound_rising_ttl) == len(sound_pc_timestamp), "sound TTL and sound PC are not the same length"




    print(sound_ttl)

if __name__ == "__main__":
    main()

