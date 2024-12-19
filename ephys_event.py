import numpy as np
import os
import pandas as pd
import sys
import time
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mat73

def pca_on_clusters(clusters):
    pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
    principal_components = pca.fit_transform(clusters)

    # Plot the principal components
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', marker='o')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Normalized Spike Counts')
    plt.show()

    # Identify the most relevant cluster (the one with the highest variance in the first principal component)
    most_relevant_cluster_idx = np.argmax(np.abs(pca.components_[0]))
    print(f"The most relevant cluster is cluster {most_relevant_cluster_idx}")

    return np.argsort(-np.abs(pca.components_[0]))

def plot_event_ephys(event_type, behavior_event, clusters, spikeTimes, time_window):
    df_event = behavior_event[behavior_event["event_name"] == f"{event_type}"]
    df_event.reset_index(drop=True, inplace=True)
    # if event_type == "L":
        
    
    # time_window = 0.5   # 0.5 seconds before and after each event
    bin_size = 0.025  # 25ms bins
    time_bins = np.arange(-time_window, time_window + bin_size, bin_size)

    # Initialize the spike count array
    event_num = len(df_event)
    cluster_num = len(clusters)
    time_bin_num = len(time_bins) - 1

    # event_num * cluster_num * time_bin (25ms bin from -0.5s to 0.5s)
    spike_count_array = np.zeros((event_num, cluster_num, time_bin_num))

    # Iterate over each event
    for event_idx, event in df_event.iterrows():
        print(f"Processing event {event_idx + 1} of {event_num}...")
        event_time = event["event_ephys_timestamp"]
        
        # Iterate over each cluster
        for cluster_idx, cluster in enumerate(clusters):     
                   
            # Convert spike times to seconds
            cluster = np.array(cluster).astype(int)
            # print("Cluster:", cluster)
            spike_times = spikeTimes[cluster] * 50 / 1e6  #/20kHz sampling rate
            
            # Count spikes in each time bin
            for bin_idx in range(time_bin_num):
                start_time = event_time + time_bins[bin_idx]
                end_time = event_time + time_bins[bin_idx + 1]
                spike_count = np.sum((spike_times >= start_time) & (spike_times < end_time))
                spike_count_array[event_idx, cluster_idx, bin_idx] = spike_count


    normalized_spike_count_array = np.mean(spike_count_array, axis=0)
    
    
    # spontaneous firing rate
    spontaneous_spike_count_array = normalized_spike_count_array[:, :41] # use the first 12 bins for spontaneous firing rate
    # TODO make the time window for spontaneous firing rate a parameter
    
    # normalization across rows(neurons)
    row_means = np.mean(spontaneous_spike_count_array, axis=1)
    print("Mean across rows:", row_means)

    # Calculate the standard deviation across each row (axis=1)
    row_stds = np.std(spontaneous_spike_count_array, axis=1)
    print("Standard deviation across rows:", row_stds)

    # Z-score normalization: (array - row_mean) / row_std
    z_score_rows = (normalized_spike_count_array - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]
    print("\nZ-score across rows:\n", np.shape(z_score_rows))
    
    
    # Remove the last bin edge to get the bin centers
    time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    # Create the raster plot
    plt.figure(figsize=(10, 6))
    plt.imshow(z_score_rows, aspect='auto', 
            extent=[time_bin_centers[0], time_bin_centers[-1], 0, 
                    z_score_rows.shape[0]], cmap='viridis')
    plt.colorbar(label='Spike Count')
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster')
    if event_type == "S":
        plt.title('Raster Plot of Normalized Spike Counts for Sound')
        
    elif event_type == "R":
        plt.title('Raster Plot of Normalized Spike Counts for Reward')
        plt.savefig('reward_response_ungrouped.png')
    elif event_type == "V":
        plt.title('Raster Plot of Normalized Spike Counts for Vacuum')
    # plt.title('Raster Plot of Normalized Spike Counts')
    plt.show()
    

    
    return z_score_rows



event_file = "/mnt/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min/session_analytics/BehaviorEvents.parquet"
cluster_file = "/mnt/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min/session_analytics/ephys_829_res.mat"

behavior_event = pd.read_parquet(event_file)

# Convert timestamps to seconds
behavior_event["event_ephys_timestamp"] = behavior_event["event_ephys_timestamp"] / 1e6


    
cluster_mat = mat73.loadmat(cluster_file)
clusters = cluster_mat["spikesByCluster"] # This is only the index for spikeTimes (spikeTimes(spikebyCluster{1,2 3 4....number of neurons}(:,:))) =this gives you the actual spike times
spikeTimes= cluster_mat["spikeTimes"]

print("Cluster size:", len(clusters))

# lick_raster = plot_event_ephys("L", behavior_event, clusters, time_window=0.5)

sound_raster = plot_event_ephys("S", behavior_event, clusters, spikeTimes, time_window=2)

reward_raster = plot_event_ephys("R", behavior_event, clusters, spikeTimes, time_window=2)

# Save the Z-score array into a MATLAB-compatible .mat file
scipy.io.savemat('z_score_array.mat', {'z_score': reward_raster})

vacuum_raster = plot_event_ephys("V", behavior_event, clusters, spikeTimes, time_window=0.5)
