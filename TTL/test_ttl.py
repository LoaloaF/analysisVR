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
    # folder = "/mnt/SpatialSequenceLearning/Unprocessed/2024-08-26_16-05-17_active/"
    folder = "/mnt/SpatialSequenceLearning/Unprocessed/2024-09-04_14-46-25_active/"
    # folder = "/mnt/SpatialSequenceLearning/Unprocessed/2024-08-26_12-57-39_active/"
    
    
    ephys_fname = "ephys_output.raw.h5"
    # behavior_fname = "behavior_2024-08-14_14-31_dummyAnimal_P0800_LinearTrack_3min.hdf5"
    # behavior_fname = "2024-08-16_14-41_dummyAnimal_P0800_LinearTrack_2min.hdf5"
    # behavior_fname = "2024-08-26_16-05_dummyAnimal_P0800_LinearTrack_0min.hdf5"
    behavior_fname = "2024-09-04_14-46_dummyAnimal_P0800_LinearTrack.xlsx_4min.hdf5"
    # behavior_fname = "2024-08-26_12-57_dummyAnimal_P0800_LinearTrack_0min.hdf5"

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
    # facecam_data = pd.read_hdf(behavioral_fullfname, key="facecam_packages")
    trial_data = pd.read_hdf(behavioral_fullfname, key="unity_trial")
    facecam_data = None
    return ephys_data, event_data, frame_data, ball_data, facecam_data, trial_data

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

def clean_ttl_data(data, threshold=10):
    filtered_data = [data[0]]
    for i in range(1, len(data)):
        if data[i] - data[i - 1] >= threshold:
            filtered_data.append(data[i])
    return np.array(filtered_data)

def main():

    ephys_data, event_data, frame_data, ball_data, facecam_data, trial_data = get_data()

    ball_ttl = ephys_data[['time', 'bit0']]
    ball_rising_ttl, ball_falling_ttl = detect_edges(ball_ttl, "bit0")

    if (ball_ttl["bit0"].iloc[0] == 1):
        ball_rising_ttl = np.insert(ball_rising_ttl, 0, ball_ttl["time"].iloc[0])
        
    ball_pc_timestamp = np.array(ball_data["ballvelocity_pc_timestamp"])
    ball_portenta_timestamp = np.array(ball_data["ballvelocity_portenta_timestamp"])

    ball_rising_ttl_norm = (ball_rising_ttl - ball_rising_ttl[0])*50
    ball_pc_timestamp_norm = ball_pc_timestamp - ball_pc_timestamp[0]
    ball_portenta_timestamp_norm = ball_portenta_timestamp - ball_portenta_timestamp[0]

    print("Ball TTL: ", len(ball_rising_ttl))
    print("Ball PC: ", len(ball_pc_timestamp))

    
    # ball_rising_ttl_norm = ball_rising_ttl_norm * 50

    # comparision_plot(ball_rising_ttl_norm[:1000], ball_pc_timestamp_norm[:1000])
    # comparision_plot(ball_rising_ttl_norm[-1000:], ball_pc_timestamp_norm[-1002:-2])
    # plt.plot(ball_pc_timestamp_norm[0:-2] - ball_rising_ttl_norm, label='Rising TTL')
    # plt.show()






    frame_ttl = ephys_data[['time', 'bit2']]
    frame_rising_ttl, frame_falling_ttl = detect_edges(frame_ttl, "bit2")
    combined_ttl = np.concatenate((frame_rising_ttl, frame_falling_ttl))
    combined_ttl = np.sort(combined_ttl)
    # clean the jitter
    combined_ttl = clean_ttl_data(combined_ttl)
    frame_pc_timestamp = np.array(frame_data["frame_pc_timestamp"])
    # take every 4th frame
    # frame_pc_timestamp = frame_pc_timestamp[::8]
    
    frame_ttl_norm = (combined_ttl[1:] - combined_ttl[1])*50
    frame_pc_timestamp_norm = frame_pc_timestamp[1:] - frame_pc_timestamp[1]
    
    print("Frame TTL: ", len(frame_ttl_norm))
    print("Frame PC: ", len(frame_pc_timestamp_norm))
    
    
    # Calculate differences between consecutive points
    diff_time = frame_ttl_norm - frame_pc_timestamp_norm[:len(frame_ttl_norm)]
    diffs = np.diff(diff_time)
    diffs2 = diff_time[2:] - diff_time[:-2]

    # Find points where the difference is larger than 30000
    indices1 = np.where(diffs > 30000)[0]
    indices2 = np.where(diffs2 > 30000)[0]
    
    indices1 += 1
    indices2 += 2
    indices = np.unique(np.concatenate((indices1, indices2, indices2-1)))
    patch_num = 0
    for indice in indices:
        time_diff_ttl = frame_ttl_norm[indice] - frame_ttl_norm[indice-1]
        time_diff_pc = frame_pc_timestamp_norm[indice + patch_num] - frame_pc_timestamp_norm[indice-1+patch_num]
        
        gap_num_ttl = round(time_diff_ttl/16666)
        gap_num_pc = round(time_diff_pc/16666)
        if time_diff_ttl < 20000 or gap_num_ttl == gap_num_pc:
            continue
        else:            
            for each_gap in range(gap_num_ttl-1):
                inter_time = frame_ttl_norm[indice-1] + time_diff_ttl/gap_num_ttl*(each_gap+1)
                inter_time_ephys = inter_time/50 + combined_ttl[1]
                time_diffs = (ephys_data['time'] - inter_time_ephys).abs()
                closest_index = time_diffs.idxmin()
                closest_time = ephys_data.loc[closest_index]["time"]
                frame_ttl_norm = np.insert(frame_ttl_norm, -1, (closest_time- combined_ttl[1])*50)
            
            print(f"For indice {indice}, the patch_num is {gap_num_ttl-1} and the time difference is {time_diff_ttl}")
            patch_num += gap_num_ttl - 1
    
    print("Patch Num: ", patch_num)
    frame_ttl_norm = np.sort(frame_ttl_norm)

    print("Frame TTL: ", len(frame_ttl_norm))
    print("Frame PC: ", len(frame_pc_timestamp_norm))
   
    plt.figure()
    plt.plot(frame_pc_timestamp_norm[:len(frame_ttl_norm)]-frame_ttl_norm)
    plt.show() 
    
    plt.figure()
    plt.plot(frame_ttl["time"],frame_ttl["bit2"])
    plt.show()
    
    plt.figure()
    bins = np.arange(0, 100000, 50)
    plt.hist(np.diff(frame_ttl_norm), bins=bins, label='Rising TTL')
    plt.yscale('log')
    plt.show()

    plt.figure()
    bins = np.arange(0, 100000, 50)
    plt.hist(np.diff(frame_pc_timestamp_norm), bins=bins, label='Rising TTL')
    plt.yscale('log')
    plt.show()



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

