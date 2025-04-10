import numpy as np
import os
import pandas as pd
import sys
import time
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mat73
from scipy.stats import zscore


def normalize_spikes(spikes, time_window):
    avg_spikes = np.mean(spikes, axis=0)
    
    # spontaneous_spikes = avg_spikes[:, :time_window] # use the first window for spontaneous firing rate

    # row_means = np.mean(spontaneous_spikes, axis=1)
    # row_stds = np.std(spontaneous_spikes, axis=1)

    # norm_spikes = (avg_spikes - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]
    norm_spikes = zscore(avg_spikes, axis=1)

    return norm_spikes
    
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

def plot_event_ephys(event_type, behavior_event, cluster_spikeTimes, time_window):
    df_event = behavior_event[behavior_event["event_name"] == f"{event_type}"]
    df_event.reset_index(drop=True, inplace=True)
    # if event_type == "L":
        
    # time_window = 0.5   # 0.5 seconds before and after each event
    bin_size = 0.025  # 25ms bins
    time_bins = np.arange(-time_window, time_window + bin_size, bin_size)

    # Initialize the spike count array
    event_num = len(df_event)
    cluster_num = len(cluster_spikeTimes)
    time_bin_num = len(time_bins) - 1

    # event_num * cluster_num * time_bin (25ms bin from -0.5s to 0.5s)
    spike_count_array = np.zeros((event_num, cluster_num, time_bin_num))

    # Iterate over each event
    for event_idx, event in df_event.iterrows():
        print(f"Processing event {event_idx + 1} of {event_num}...")
        event_time = event["event_ephys_timestamp"]
        
        # Iterate over each cluster
        for cluster_idx, cluster in enumerate(cluster_spikeTimes):     
                   
            # Convert spike times to seconds
            cluster = np.array(cluster).astype(int)
            # print("Cluster:", cluster)
            spike_times = cluster * 50 / 1e6  #/20kHz sampling rate
            
            # Count spikes in each time bin
            for bin_idx in range(time_bin_num):
                start_time = event_time + time_bins[bin_idx]
                end_time = event_time + time_bins[bin_idx + 1]
                spike_count = np.sum((spike_times >= start_time) & (spike_times < end_time))
                spike_count_array[event_idx, cluster_idx, bin_idx] = spike_count


    normalized_spikes = normalize_spikes(spike_count_array, time_window)
    
    
    # Remove the last bin edge to get the bin centers
    time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    # Create the raster plot
    plt.figure(figsize=(10, 6))
    plt.imshow(normalized_spikes, aspect='auto', 
            extent=[time_bin_centers[0], time_bin_centers[-1], 0, 
                    normalized_spikes.shape[0]], cmap='viridis')
    plt.colorbar(label='Spike Count')
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster')
    plt.title(f'Raster Plot of Normalized Spike Counts for {event_type}')
    plt.show()
    
    return normalized_spikes


def plot_unity_ephys(unity_data, cluster_spikeTimes, bin_size):
    trial_ids = unity_data["trial_id"].unique()
    trial_ids = trial_ids[trial_ids != -1]
    
    trial_num = len(trial_ids)
    cluster_num = len(cluster_spikeTimes)
    time_bin_num = 86

    spike_count_array = np.zeros((trial_num, cluster_num, time_bin_num))
        
    for trial_idx in trial_ids:
        print(f"Processing trial {trial_idx} of {trial_num}...")
        df_trial = unity_data[unity_data["trial_id"] == trial_idx]
        df_trial = df_trial[["frame_z_position", "frame_ephys_timestamp"]]

        # Initialize an empty list to store the results
        results = []

        # Iterate through the z positions and timestamps
        for start_pos in range(-169, 261, bin_size):
            end_pos = start_pos + bin_size
            bin_data = df_trial[(df_trial["frame_z_position"] >= start_pos) & (df_trial["frame_z_position"] < end_pos)]
            
            if not bin_data.empty:
                start_time = bin_data["frame_ephys_timestamp"].iloc[0]
                end_time = bin_data["frame_ephys_timestamp"].iloc[-1]
                time_interval = end_time - start_time
                results.append([start_pos, start_time, end_time, time_interval])

        # Convert the results to a DataFrame
        df_bins = pd.DataFrame(results, columns=["start_position", "start_time", "end_time", "time_interval"])

        for cluster_idx, cluster in enumerate(cluster_spikeTimes):     
            # Convert spike times to seconds
            spike_times = cluster * 50 / 1e6  #/20kHz sampling rate
            
            # Count spikes in each time bin
            for bin_idx, bin_data in df_bins.iterrows():
                start_time = bin_data["start_time"]
                end_time = bin_data["end_time"]
                spike_count = np.sum((spike_times >= start_time) & (spike_times < end_time))
                spike_count = spike_count / bin_data["time_interval"]
                spike_count_array[trial_idx-1, cluster_idx, bin_idx] = spike_count

    norm_spikes = normalize_spikes(spike_count_array, 5)

    start_positions = np.arange(-169, 261, bin_size)
    # Create the plot
    # plt.figure(figsize=(12, 8))
    # plt.imshow(norm_spikes, aspect='auto', extent=[start_positions[0], start_positions[-1] + bin_size, 0, norm_spikes.shape[0]], cmap='viridis')
    # plt.colorbar(label='Average Spiking Rate')
    # plt.xlabel('Position Bins (cm)')
    # plt.ylabel('Cluster')
    # plt.title('Average Spiking Rate of Clusters on a Linear Track')
    # plt.xticks(start_positions)
    
    max_firing_positions = np.argmax(norm_spikes, axis=1)
    # Sort the clusters based on the position of maximum firing
    sorted_indices = np.argsort(max_firing_positions)
    sorted_norm_spikes = norm_spikes[sorted_indices, :]

    # Create the space coding plot
    plt.figure(figsize=(12, 8))
    plt.imshow(sorted_norm_spikes, aspect='auto', extent=[start_positions[0], start_positions[-1] + bin_size, 0, sorted_norm_spikes.shape[0]], cmap='viridis')
    plt.colorbar(label='Average Spiking Rate')
    plt.xlabel('Position Bins (cm)')
    plt.ylabel('Cluster (sorted by firing position)')
    plt.title('Space Coding Plot of Clusters on a Linear Track')
    plt.xticks(start_positions)
    
    
    # reward 2
    plt.axvline(x=230, color='lightblue', linestyle='--', linewidth=1)
    plt.axvline(x=170, color='lightblue', linestyle='--', linewidth=1)
    
    # reward 1
    plt.axvline(x=110, color='purple', linestyle='--', linewidth=1)
    plt.axvline(x=50, color='purple', linestyle='--', linewidth=1)
    
    # cue
    plt.axvline(x=-130, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=10, color='black', linestyle='--', linewidth=1)
    
    plt.show()
    
    return norm_spikes


def extract_ephys(cluster_file):
    cluster_mat = mat73.loadmat(cluster_file)
    clusters = cluster_mat["spikesByCluster"] # This is only the index for spikeTimes (spikeTimes(spikebyCluster{1,2 3 4....number of neurons}(:,:))) =this gives you the actual spike times
    spikeTimes= cluster_mat["spikeTimes"]
    print(len(spikeTimes))

    cluster_spikeTimes = []
    for each_cluster in clusters:
        cluster_spikeTimes.append(spikeTimes[each_cluster[0].astype(int)])
    print("Cluster size:", len(cluster_spikeTimes))

    
    # alternative     
    clusters = cluster_mat["spikeClusters"] 
    spikeTimes = cluster_mat["spikeTimes"] * 50 # TODO: check if this is correct to multiply 50
    cluster_sites = cluster_mat["clusterSites"]

    spikes = pd.DataFrame({
    "cluster_id": clusters,
    "spike_time": spikeTimes
    })
    
    print("Unique cluster IDs in spikes:", spikes['cluster_id'].unique())
    print("Number of cluster sites:", len(cluster_sites))
    print(f"Ckluster sites cluster_sites {cluster_sites}")
    # exit()
    
    # Create a mapping from cluster_id to site_id using the actual cluster_id values
    cluster_to_site = {cluster_id: site_id for cluster_id, site_id in zip(clusters, cluster_sites)}

    # Map the site_id values to the spikes DataFrame
    spikes['site_id'] = spikes['cluster_id'].map(cluster_to_site)

    
    print(spikes)
    print(spikes['site_id'].isna().sum())
    
    return cluster_spikeTimes


def get_Spikes(session_fullfname):
    session_dir = os.path.dirname(session_fullfname)
    analytics_dir = os.path.join(session_dir, "session_analytics")
    
    ephys_res = [file for file in os.listdir(analytics_dir) if "ephys" in file and file.endswith("_res.mat")]
    
    if not ephys_res:
        # Logger().logger.warning(f"No ephys file found in {analytics_dir}")
        return None
    
    ephys_res = os.path.join(analytics_dir, ephys_res[0])
    ephys_res_mat = mat73.loadmat(ephys_res)

    # TODO: include more fields to this parquet
    clusters = ephys_res_mat["spikeClusters"] 
    spikeTimes = ephys_res_mat["spikeTimes"] * 50 # TODO: check if this is correct to multiply 50
    cluster_sites = ephys_res_mat["clusterSites"]

    spikes = pd.DataFrame({
    "cluster_id": clusters,
    "spike_time": spikeTimes
    })
    
    # Create a mapping from cluster to site
    cluster_to_site = {cluster_id: site_id for cluster_id, site_id in enumerate(cluster_sites)}
    spikes['site_id'] = spikes['cluster_id'].map(cluster_to_site)

    return spikes



# nas_dir = "/mnt/SpatialSequenceLearning/"
nas_dir = "/mnt/SpatialSequenceLearning"
nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
session_dir = "RUN_rYL006/rYL006_P1100/2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min"

event_file = os.path.join(nas_dir, session_dir, "session_analytics/BehaviorEvents.parquet")
unity_file = os.path.join(nas_dir, session_dir, "session_analytics/UnityFramewise.parquet")
cluster_file = os.path.join(nas_dir, session_dir, "session_analytics/ephys_829_res.mat")

behavior_event = pd.read_parquet(event_file)
unity_data = pd.read_parquet(unity_file)
behavior_event["event_ephys_timestamp"] = behavior_event["event_ephys_timestamp"] / 1e6
unity_data["frame_ephys_timestamp"] = unity_data["frame_ephys_timestamp"] / 1e6


cluster_spikeTimes = extract_ephys(cluster_file)
exit()
print(cluster_spikeTimes)
print(len(cluster_spikeTimes))
[print(len(cluster)) for cluster in cluster_spikeTimes]
raster = plot_event_ephys("R", behavior_event, cluster_spikeTimes, time_window=2)
exit()

# # lick_raster = plot_event_ephys("L", behavior_event, clusters, time_window=0.5)
# sound_raster = plot_event_ephys("S", behavior_event, cluster_spikeTimes, time_window=2)
# reward_raster = plot_event_ephys("R", behavior_event, cluster_spikeTimes, time_window=2)
# # Save the Z-score array into a MATLAB-compatible .mat file
# scipy.io.savemat('z_score_array.mat', {'z_score': reward_raster})
# vacuum_raster = plot_event_ephys("V", behavior_event, cluster_spikeTimes, time_window=0.5)

unity_raster = plot_unity_ephys(unity_data, cluster_spikeTimes, 5)


