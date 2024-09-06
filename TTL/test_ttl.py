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
    folder = "/mnt/SpatialSequenceLearning/Unprocessed/2024-09-06_14-53-48_active/"
    # folder = "/mnt/SpatialSequenceLearning/Unprocessed/2024-08-26_12-57-39_active/"
    
    
    ephys_fname = "ephys_output.raw.h5"
    # behavior_fname = "behavior_2024-08-14_14-31_dummyAnimal_P0800_LinearTrack_3min.hdf5"
    # behavior_fname = "2024-08-16_14-41_dummyAnimal_P0800_LinearTrack_2min.hdf5"
    # behavior_fname = "2024-08-26_16-05_dummyAnimal_P0800_LinearTrack_0min.hdf5"
    behavior_fname = "2024-09-06_14-54_dummyAnimal_P0800_LinearTrack.xlsx_17min.hdf5"
    # behavior_fname = "2024-08-26_12-57_dummyAnimal_P0800_LinearTrack_0min.hdf5"

    ephys_fullfname = os.path.join(folder, ephys_fname)
    behavioral_fullfname = os.path.join(folder, behavior_fname)

    with h5py.File(ephys_fullfname, 'r') as file:
        # ephys_bits = file['bits'][:]
        ephys_bits = file['bits'][:]

    # convert the value into single bits and create time series
    msg = np.array([(a[0],a[1]) for a in ephys_bits])

    ephys_data = pd.DataFrame(msg, columns=['time', 'value'])

    for column_id in range(8):
        ephys_data[f'bit{column_id}'] = (ephys_data['value'] & (1 << column_id))/2**column_id
        ephys_data[f'bit{column_id}'] = ephys_data[f'bit{column_id}'].astype(int)
    
    frame_data = pd.read_hdf(behavioral_fullfname, key="unity_frame")
    ball_data = pd.read_hdf(behavioral_fullfname, key="ballvelocity")
    
    try:
        event_data = pd.read_hdf(behavioral_fullfname, key="event")
    except:
        event_data = None
    try:
        facecam_data = pd.read_hdf(behavioral_fullfname, key="facecam_packages")
    except:
        facecam_data = None
        
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

def clean_ttl_data(data, threshold=10):
    filtered_data = [data[0]]
    for i in range(1, len(data)):
        if data[i] - data[i - 1] >= threshold:
            filtered_data.append(data[i])
    return np.array(filtered_data)

def main():

    ephys_data, event_data, frame_data, ball_data, facecam_data = get_data()

    print("-------------------")
    ball_ttl = ephys_data[['time', 'bit0']]
    ball_rising_ttl, ball_falling_ttl = detect_edges(ball_ttl, "bit0")

    if (ball_ttl["bit0"].iloc[0] == 1):
        ball_rising_ttl = np.insert(ball_rising_ttl, 0, ball_ttl["time"].iloc[0])
        
    ball_pc_timestamp = np.array(ball_data["ballvelocity_pc_timestamp"])

    ball_rising_ttl_norm = (ball_rising_ttl - ball_rising_ttl[0])*50
    ball_pc_timestamp_norm = ball_pc_timestamp - ball_pc_timestamp[0]

    print("Ball TTL: ", len(ball_rising_ttl_norm))
    print("Ball PC: ", len(ball_pc_timestamp_norm))
    
    plt.figure()
    plt.plot(ball_rising_ttl_norm - ball_pc_timestamp_norm)
    plt.title('Difference between TTL and PC Timestamp: TTL - PC')
    plt.show() 
    
    ball_data["ballvelocity_ephys_timestamp"] = ball_rising_ttl_norm/50 + ball_rising_ttl[0]

    print("-------------------")
    frame_ttl = ephys_data[['time', 'bit2']]
    frame_rising_ttl, frame_falling_ttl = detect_edges(frame_ttl, "bit2")
    combined_ttl = np.concatenate((frame_rising_ttl, frame_falling_ttl))
    combined_ttl = np.sort(combined_ttl)
    # clean the jitter
    combined_ttl = clean_ttl_data(combined_ttl)
    frame_pc_timestamp = np.array(frame_data["frame_pc_timestamp"])
    
    frame_ttl_norm = (combined_ttl[1:] - combined_ttl[1])*50
    frame_pc_timestamp_norm = frame_pc_timestamp[1:] - frame_pc_timestamp[1]
    
    print("Before Patching:")
    print("Frame TTL: ", len(frame_ttl_norm))
    print("Frame PC: ", len(frame_pc_timestamp_norm))
    
    current_idx = 0
    number_of_inserts = 0
    number_of_deletes = 0
    modify_idx_list = []
    threshold = 6000
    while (current_idx < len(frame_pc_timestamp_norm)):
        time_diff = frame_ttl_norm[current_idx] - frame_pc_timestamp_norm[current_idx]
        if time_diff < threshold and time_diff > -threshold:
            current_idx += 1
        elif time_diff >= threshold:
            insert_time = frame_pc_timestamp_norm[current_idx]/50 + combined_ttl[1]
            time_diffs = (ephys_data['time'] - insert_time).abs()
            closest_index = time_diffs.idxmin()
            closest_time = ephys_data.loc[closest_index]["time"]
            frame_ttl_norm = np.insert(frame_ttl_norm, current_idx, (closest_time- combined_ttl[1])*50)
            number_of_inserts += 1
            modify_idx_list.append(current_idx)
        elif time_diff <= -threshold:
            frame_ttl_norm = np.delete(frame_ttl_norm, current_idx)
            number_of_deletes += 1
            modify_idx_list.append(current_idx)
    
    print("Number of Inserts: ", number_of_inserts)
    print("Number of Deletes: ", number_of_deletes)
   
    print("After Patching:")
    print("Frame TTL: ", len(frame_ttl_norm))
    print("Frame PC: ", len(frame_pc_timestamp_norm))
   
    plt.figure()
    plt.plot(frame_ttl_norm - frame_pc_timestamp_norm[:len(frame_ttl_norm)])
    plt.title('Difference between TTL and PC Timestamp after patching: TTL - PC')
    plt.show() 
    
    frame_data = frame_data.iloc[1:]
    frame_data["frame_ephys_timestamp"] = frame_ttl_norm/50 + combined_ttl[1]
    
    if facecam_data is None:
        print("-------------------")
        print("No Facecam Data")
    else:
        facecam_ttl = ephys_data[['time', 'bit3']]
        facecam_rising_ttl, facecam_falling_ttl = detect_edges(facecam_ttl, "bit3")
        
        if (len(facecam_rising_ttl) == 0):
            print("No Facecam TTL")
        else:
            facecam_pc_timestamp = np.array(facecam_data["facecam_image_pc_timestamp"])
            facecam_rising_ttl_norm = (facecam_rising_ttl - facecam_rising_ttl[0])*50
            facecam_pc_timestamp_norm = facecam_pc_timestamp - facecam_pc_timestamp[0]

            print("Facecam TTL: ", len(facecam_rising_ttl_norm))
            print("Facecam PC: ", len(facecam_pc_timestamp_norm))
            
            plt.figure()
            plt.plot(facecam_rising_ttl_norm - facecam_pc_timestamp_norm)
            plt.title('Difference between TTL and PC Timestamp after patching: TTL - PC')
            plt.show() 
            
            facecam_data["facecam_image_ephys_timestamp"] = facecam_rising_ttl_norm/50 + facecam_rising_ttl[0]
        
    print("-------------------")
    if event_data is None:
        print("No Event Data")
    else:
        lick_ttl = ephys_data[['time', 'bit5']]
        lick_rising_ttl, lick_falling_ttl = detect_edges(lick_ttl, "bit5")
        
        if (len(lick_rising_ttl) == 0):
            print("No Lick TTL")
        else:
            lick_pc_timestamp = np.array(event_data[event_data["event_name"]=="L"]["event_pc_timestamp"])
            lick_rising_ttl_norm = (lick_rising_ttl - lick_rising_ttl[0])*50
            lick_pc_timestamp_norm = lick_pc_timestamp - lick_pc_timestamp[0]

            print("Lick TTL: ", len(lick_rising_ttl_norm))
            print("Lick PC: ", len(lick_pc_timestamp_norm))
            
            plt.figure()
            plt.plot(lick_rising_ttl_norm - lick_pc_timestamp_norm)
            plt.title('Difference between TTL and PC Timestamp after patching: TTL - PC')
            plt.show() 
            
            event_data.loc[event_data["event_name"]=="L", "event_ephys_timestamp"] = lick_rising_ttl_norm/50 + lick_rising_ttl[0]

        print("-------------------")
        reward_ttl = ephys_data[['time', 'bit6']]
        reward_rising_ttl, reward_falling_ttl = detect_edges(reward_ttl, "bit6")
        
        if (len(reward_rising_ttl) == 0):
            print("No Reward TTL")
        else:
            reward_pc_timestamp = np.array(event_data[event_data["event_name"]=="R"]["event_pc_timestamp"])
            reward_rising_ttl_norm = (reward_rising_ttl - reward_rising_ttl[0])*50
            reward_pc_timestamp_norm = reward_pc_timestamp - reward_pc_timestamp[0]

            print("Reward TTL: ", len(reward_rising_ttl_norm))
            print("Reward PC: ", len(reward_pc_timestamp_norm))
            
            plt.figure()
            plt.plot(reward_rising_ttl_norm - reward_pc_timestamp_norm)
            plt.title('Difference between TTL and PC Timestamp after patching: TTL - PC')
            plt.show()
            
            event_data.loc[event_data["event_name"]=="R", "event_ephys_timestamp"] = reward_rising_ttl_norm/50 + reward_rising_ttl[0]

        print("-------------------")
        sound_ttl = ephys_data[['time', 'bit7']]
        sound_rising_ttl, sound_falling_ttl = detect_edges(sound_ttl, "bit7")
        
        if (len(sound_rising_ttl) == 0):
            print("No Sound TTL")
        else:
            sound_pc_timestamp = np.array(event_data[event_data["event_name"]=="S"]["event_pc_timestamp"])
            sound_rising_ttl_norm = (sound_rising_ttl - sound_rising_ttl[0])*50
            sound_pc_timestamp_norm = sound_pc_timestamp - sound_pc_timestamp[0]

            print("Sound TTL: ", len(sound_rising_ttl_norm))
            print("Sound PC: ", len(sound_pc_timestamp_norm))
            
            plt.figure()
            plt.plot(sound_rising_ttl_norm - sound_pc_timestamp_norm)
            plt.title('Difference between TTL and PC Timestamp after patching: TTL - PC')
            plt.show()
            
            event_data.loc[event_data["event_name"]=="S", "event_ephys_timestamp"] = sound_rising_ttl_norm/50 + sound_rising_ttl[0]
        

if __name__ == "__main__":
    main()

