import os
import numpy as np
import pandas as pd
import h5py

from CustomLogger import CustomLogger as Logger
import analytics_processing.sessions_from_nas_parsing as sp
import analytics_processing.modality_loading as ml
from analytics_processing.analytics_constants import device_paths

from dashsrc.plot_components.plot_utils import make_discr_cluster_id_cmap

# from ../../ephysVR.git
from mea1k_modules.mea1k_raw_preproc import mea1k_raw2decompressed_dat_file

def _handle_session_ephys(session_fullfname, mode=False, exclude_shanks=None):
    L = Logger()
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    session_dir = os.path.dirname(session_fullfname)
    animal_name = session_name.split("_")[2]
    
    # check if a dat file already exists
    dat_fnames = [f for f in os.listdir(session_dir) if f.endswith(".dat")]
    dat_sizes = np.array([os.path.getsize(os.path.join(session_dir, f)) for f in dat_fnames])

    if any(dat_sizes>0):
        L.logger.debug(f"De-compressed ephys files found: {dat_fnames}")
        if mode != "recompute":
            L.logger.debug(f"Skipping...")
            return
        else:
            L.logger.debug(f"Recomputing...")
        
    
    if not os.path.exists(os.path.join(session_dir, "ephys_output.raw.h5")):
        L.logger.warning(f"No raw ephys recordings found for {session_name}")
        return

    # decompress the raw output of mea1k and convert to uV int16 .dat file
    # also, save a mapping csv file for identifying the rows in the dat file
    mea1k_raw2decompressed_dat_file(session_dir, 'ephys_output.raw.h5', 
                                    session_name=session_name,
                                    animal_name=animal_name,
                                    convert2uV=True,
                                    subtract_dc_offset=True,
                                    write_neuroscope_xml=True,
                                    write_probe_file=True,
                                    replace_with_curated_xml=False,
                                    exclude_shanks=exclude_shanks)
    
def postprocess_ephys(**kwargs):
    L = Logger()
    exclude_shanks = kwargs.pop("exclude_shanks")
    mode = kwargs.pop("mode")
    
    sessionlist_fullfnames, ids = sp.sessionlist_fullfnames_from_args(**kwargs)

    aggr = []
    for i, (session_fullfname, identif) in enumerate(zip(sessionlist_fullfnames, ids)):
        L.logger.info(f"Processing {identif} ({i+1}/{len(ids)}) "
                      f"{os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        
        # create the dat file 
        _handle_session_ephys(session_fullfname, mode, exclude_shanks)
        
def _unpack_JRC_output(session_fullfname, get_spikes=False, get_cluster_metadata=False):
    L = Logger()
    
    # excluseive or check
    if not get_spikes and not get_cluster_metadata or (get_spikes and get_cluster_metadata):
        raise ValueError("Exactly one of `get_spikes` or `get_cluster_metadata` must be True")
    
    nas = device_paths()[0]
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    animal_name = session_name.split("_")[2]
    ss_dirnames = pd.read_csv(os.path.join(nas, "devices", "animal_meta_info.csv"), 
                              index_col='animal_name').loc[animal_name, 'ss_dirnames'].split(",")
    traces, mapping = ml.session_modality_from_nas(session_fullfname, key='ephys_traces')
    if mapping is None:
        return None
    print(f"Mapping:\n{mapping}")   

    aggr_over_ss_batches = []
    for ss_i, ss_dirname in enumerate(ss_dirnames):
        ss_fulldir = os.path.join(nas, f"RUN_{animal_name}", "concatenated_ss", ss_dirname)
        L.logger.debug(f"Processing spikes from {ss_fulldir} for {session_name}")
        
        ss_session_lengths = pd.read_csv(os.path.join(ss_fulldir, "concat_session_lengths.csv"))
        ss_session_names = ss_session_lengths['name'].apply(lambda x: x[:x.find("min_")+3]).values
        L.logger.debug(f"Containes spike sorted sessions:\n{ss_session_names}")
        
        
        entry_i = np.where(ss_session_names == session_name)[0]
        if len(entry_i) == 0:
            L.logger.warning(f"Session {session_name} not spike sorted in"
                             f" {os.path.basename(ss_fulldir)}")
            continue
        session_from_smple = ss_session_lengths.iloc[entry_i[0]-1].loc["nsamples_cum"] if entry_i[0] != 0 else 0
        session_to_smple = ss_session_lengths.iloc[entry_i[0]].loc["nsamples_cum"]
        
        ss_res_fullfname = os.path.join(ss_fulldir, 'concat_res.mat')
        if not os.path.exists(ss_res_fullfname):
            raise FileNotFoundError(f"Did not find _res.mat file in {ss_fulldir}")
        
        # print_hdf5_keys(ss_res_fullfname)
        
        if get_cluster_metadata:
            with h5py.File(ss_res_fullfname, 'r') as f:
                
                cluster_notes_data = []
                for note_raw in [f[obj_ref][()] for obj_ref in f['clusterNotes'][0]]:
                    cluster_notes_data.append("".join([chr(c.item()) for c in note_raw]))
                cluster_notes_data = [n if n != '\x00\x00' else '' for n in cluster_notes_data]
                clust_id = np.sort(np.unique(f['spikeClusters'][0]))
                clust_id = clust_id[~np.isin(clust_id, (-1, 0))].astype(int)
                cluster_table = pd.DataFrame({
                    "cluster_id": clust_id +(10_000*ss_i), # make ids unique
                    "cluster_type": cluster_notes_data,
                    "unit_count": f['unitCount'][0],
                    "cluster_channel": f['clusterSites'][:, 0],
                    "cluster_id_ssbatch": clust_id,
                    "unit_snr": f['unitSNR'][0],
                    "unit_Vpp": f['unitVpp'][0],
                    "unit_isi_ratio": f['unitISIRatio'][:,0],
                    "unit_iso_dist": f['unitIsoDist'][:,0],
                    "unit_L_ratio": f['unitLRatio'][:,0],
                })
            cluster_table['session_nsamples'] = traces.shape[1]
            cluster_table['ss_batch_name'] = ss_dirname
            cluster_table['ss_batch_id'] = ss_i

            mapping_cols = ['amplifier_id', 'mea1k_el', 'pad_id', 'metal', 'shank_name', 
                            'el_pair', 'mea1k_connectivity', 'connectivity_order',
                            'shank_side', 'curated_trace', 'depth', 'shank_id']
            cluster_metad = mapping.loc[cluster_table.cluster_channel, mapping_cols].drop_duplicates()
            cluster_metad = cluster_metad.reset_index().rename({'index': 'cluster_channel',
                                                                'el_pair': 'fiber_id'}, axis=1)
            cluster_table = pd.merge(cluster_table, cluster_metad, on='cluster_channel')
            cluster_table = cluster_table[~np.isin(cluster_table.cluster_type, ('to_delete', ""))]
            aggr_over_ss_batches.append(cluster_table)
            
            L.logger.debug(f"Spike cluster table:\n{cluster_table}")

        
        elif get_spikes:
            with h5py.File(ss_res_fullfname, 'r') as f:
                spike_times = f['spikeTimes'][0]
                ss_from_spike_i = np.where((session_from_smple < spike_times))[0][0]
                ss_to_spike_i = np.where((session_to_smple < spike_times))[0]
                if len(ss_to_spike_i) != 0:
                    ss_to_spike_i = ss_to_spike_i[0]
                else:
                    # last spike before end of session
                    ss_to_spike_i = len(spike_times)-1
                L.logger.debug(f"Session samples within concatenated data conext go"
                            f" from {session_from_smple:,} to {session_to_smple:,}." 
                            f"All spike ({len(spike_times):,}) go from sample {spike_times[0]:,} to {spike_times[-1]:,}."
                            f" This session's spikes are at: [{spike_times[ss_from_spike_i]:,} ... {spike_times[ss_to_spike_i]:,}]")
                
                spike_times = spike_times[ss_from_spike_i:ss_to_spike_i]
                spike_clusters = f['spikeClusters'][0, ss_from_spike_i:ss_to_spike_i]
                spike_amps = f['spikeAmps'][0, ss_from_spike_i:ss_to_spike_i]
                spike_positions = f['spikePositions'][0, ss_from_spike_i:ss_to_spike_i]
                spike_sites = f['spikeSites'][0, ss_from_spike_i:ss_to_spike_i]
            
            spike_table = pd.DataFrame({
                "sample_id": spike_times-session_from_smple,
                "ephys_timestamp": (spike_times-session_from_smple).astype(np.uint64)*50,
                "cluster_id_ssbatch": spike_clusters.astype(int),
                "ss_batch_id": ss_i,
                "amplitude_uV": np.round(spike_amps *6.3), # convert to uV
                "channel": spike_sites-1,
                "shank": np.round(spike_positions)//1000,
            })
            # add depth
            spike_table['depth'] = mapping.depth.values[spike_table.channel]
            # remove 'to_delte' clusters (-1)
            spike_table = spike_table.loc[spike_table.cluster_id_ssbatch!=-1].reset_index(drop=True)
            # Sort the table by 'sample_id', 'cluster_id_ssbatch', and 'amplitude_uV' in descending order
            spike_table = spike_table.sort_values(by=['sample_id', 'cluster_id_ssbatch', 'amplitude_uV'], 
                                                  ascending=[True, True, False])
            # Drop duplicates, keeping the first (highest amplitude) for each group
            spike_table = spike_table.drop_duplicates(subset=['sample_id', 'cluster_id_ssbatch'], keep='first')
            # TODO redo this with a small interval around each spike
            L.logger.debug(f"Extracted spikes from {len(spike_table.cluster_id_ssbatch.unique())-1}"
                           f" unique clusters:\n{spike_table}")
            # add waveform placeholder (after logging)
            spike_table['waveform'] = [[np.int16(0)] * 20] * len(spike_table)
            aggr_over_ss_batches.append(spike_table)
            
            # feastues = np.memmap(os.path.join(ss_fulldir, 'concat_features.jrc'), dtype='float32',
            #                 mode='r').reshape((-1, l), order='F')[:, ss_from_spike_i:ss_to_spike_i]
            # raw = np.memmap(os.path.join(ss_fulldir, 'concat_filt.jrc'), dtype='int16', 
            #                 mode='r').reshape((wf_length, -1, l), 
            #                                   order='F')[:20,:, ss_from_spike_i:ss_to_spike_i]
            # chnl_i_lowest_peak = np.argsort(raw, axis=1, )
            # import matplotlib.pyplot as plt
            # for j in range(20,40):
            #     minval, lowest_i = 0, None
            #     for i in range(22):
            #         print(np.min(raw[:20, i,j]-raw[0,i,j]))
            #         plt.plot(raw[:20, i,j].T - raw[0,i,j], alpha=(i+1)/22)
            #         if np.min(raw[:20, i,j]-raw[0,i,j]) < minval:
            #             minval = np.min(raw[:20, i,j]-raw[0,i,j])
            #             lowest_i = i
            #     plt.plot(raw[:20, lowest_i,j].T - raw[0,lowest_i,j], color='k', linewidth=5, alpha=1)
            #     plt.show()
            
            # import time
            # nspikes_per_read = 100
            # spike_times_smpl = spike_table.sample_id.values
            # smpl_chunks = np.unique(spike_times_smpl)[::nspikes_per_read]
            # print(len(smpl_chunks))
            # for chunk_i, (from_spike_smple, to_spike_smple) in enumerate(zip(smpl_chunks[:-1], smpl_chunks[1:])):
            #     # to_spike_smple = spike_times_smpl[(chunk_i+1) * nspikes_per_read]
            #     time.sleep(0.1)
            #     print(chunk_i, to_spike_smple-from_spike_smple)
            #     traces_chunk = traces[:, min(0,from_spike_smple-10):to_spike_smple+10]
            #     print(traces_chunk, traces_chunk.shape)
                
            #     chunk_spikes = spike_table[(spike_table.sample_id >= from_spike_smple) &
            #                               (spike_table.sample_id < to_spike_smple)]
            #     chunk_spike_smpls = chunk_spikes.sample_id.values - from_spike_smple
            #     print(chunk_spike_smpls)
            #     wf = traces_chunk[spike_table.channel.values, chunk_spike_smpls]
            #     print(wf)
            #     print(wf.shape)
                

        L.spacer("debug")
    aggr_data = pd.concat(aggr_over_ss_batches, axis=0)
    if get_spikes:
        aggr_data.sort_values(by='sample_id', inplace=True)
    
    elif get_cluster_metadata:
        unique_ids = aggr_data.cluster_id.unique()
        renamer = {old_id: new_id+1 for new_id, old_id in enumerate(unique_ids)}
        aggr_data['cluster_id'] = aggr_data['cluster_id'].replace(renamer)
        
        # assign colors per cluster    
        map_colors = np.vectorize(make_discr_cluster_id_cmap(aggr_data['cluster_id']).get)
        aggr_data['cluster_color'] = map_colors(aggr_data['cluster_id'])
        L.logger.debug(f"Aggregation over spike sorting groups:\n{aggr_data}")
    return aggr_data

def get_SpikeClusterMetadata(session_fullfname):
    return _unpack_JRC_output(session_fullfname, get_cluster_metadata=True)
        
def get_Spikes(session_fullfname, sp_clust_metadata):
    spikes = _unpack_JRC_output(session_fullfname, get_spikes=True )
    # create a mapping between non_unique cluster ids from superate spike sorting batches to unique ids in metadata
    unq_id_renamer = (sp_clust_metadata.loc[:, ['cluster_id_ssbatch', 'ss_batch_id']].values).tolist()
    unq_id_renamer = {str(k): c_id for k, c_id in zip(unq_id_renamer, sp_clust_metadata.cluster_id)}
    
    # use the mapping to assign unique cluster ids to single spikes
    keys = [str(k) for k in spikes.loc[: , ['cluster_id_ssbatch', 'ss_batch_id']].values.tolist()]
    spikes['cluster_id'] = [unq_id_renamer.get(k, 0) for k in keys]
    
    # add cluster color for each spike
    cols = sp_clust_metadata.set_index('cluster_id').reindex(spikes['cluster_id']).cluster_color
    spikes['cluster_color'] = cols.fillna('#888888').values # needed bc of cl_id == 0 or not in metadata yet (sorted)
    spikes = spikes[np.isin(spikes.cluster_id, sp_clust_metadata.cluster_id)]
    return spikes

def get_FiringRate40msHz(spikes):
    # create a timer interval calmun from sample_id
    bin_size_us = 40_000 # in us
    spikes['bin_200ms'] = pd.cut(
        spikes['ephys_timestamp'],
        bins=pd.IntervalIndex.from_breaks(
            np.arange(0, spikes['ephys_timestamp'].max()+bin_size_us, 
                      bin_size_us, dtype='uint64'),
        )
    )
    # then group by clusterID and use the bin_200ms to count spikes
    cnt = lambda x: x['bin_200ms'].value_counts()/0.2
    fr_200ms_hz = spikes.groupby('cluster_id').apply(cnt, include_groups=False)
    # return the riting rate with time_bins on the index and cluster_id on the columns
    return fr_200ms_hz.unstack().fillna(0).T

def get_FiringRate40msZ(fr_hz):
    def zscore(x):
        return (x - x.mean()) / x.std()
    return fr_hz.apply(zscore, axis=0)

def get_FiringRateTrackbinsHz(fr_z, track_data):
    
    def trackwise_averages(fr, track_data):
        def time_bin_avg(trial_data, ):
            print(f"{trial_data.from_z_position_bin.iloc[0]:04}", end='\r')
            trial_aggr = []
            for row in range(trial_data.shape[0]):
                from_t, to_t = trial_data.iloc[row].loc[['posbin_from_ephys_timestamp', 'posbin_to_ephys_timestamp']]
                from_t = from_t - 40_000
                to_t = to_t + 40_000
                
                trial_d_track_bin_fr = fr.loc[(fr.index.left > from_t) & (fr.index.right < to_t)]
                # print(trial_data.iloc[row])
                # print(trial_d_track_bin_fr.shape, to_t-from_t, )
                # print()
                trial_aggr.append(trial_d_track_bin_fr.values)
            trial_aggr_concat = np.concatenate(trial_aggr, axis=0)
            if trial_aggr_concat.shape[0] == 0:
                # no trials in this bin
                return pd.Series(np.nan, index=fr.columns)
            m = trial_aggr_concat.mean(axis=0)
            return pd.Series(m, index=fr.columns)
                
        print("Track bin: ")
        return track_data.groupby('from_z_position_bin', ).apply(time_bin_avg)
    
    # recover index
    old_idx = fr_z.index
    fr_z = fr_z.reset_index(drop=True)
    new_idx = pd.IntervalIndex.from_breaks(
        np.arange(0, fr_z.shape[0]*40_000+1, 40_000, dtype='uint64'),
    )
    fr_z.index = new_idx
    fr_hz_averages = trackwise_averages(fr_z, track_data)
    return fr_hz_averages
    import matplotlib.pyplot as plt
    plt.imshow(fr_hz_averages.values.T, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.show()
    print(fr_hz_averages)
    

# def _rastermap_order(spikes, **kwargs):
        # spike_counts = spikes.cluster_id.value_counts().reindex(cluster_metadata.cluster_id)
        # spike_avg_fr = spike_counts / cluster_metadata.session_nsamples.values[0] *20_000
        # cluster_metadata['session_nspikes'] = spike_counts.values
        # cluster_metadata['session_avg_fr_hz'] = spike_avg_fr.values
        # return cluster_metadata
#     model = Rastermap(n_clusters=None, # None turns off clustering and sorts single neurons 
#                   n_PCs=12, # use fewer PCs than neurons
#                   locality=0.6, # some locality in sorting (this is a value from 0-1)
#                   time_lag_window=50, # use future timepoints to compute correlation
#                   grid_upsample=0, # 0 turns off upsampling since we're using single neurons
#                 ).fit(spks)
#     y = model.embedding # neurons x 1
#     isort = model.isort
    
def print_hdf5_keys(file_path, max_elements=1000):
    with h5py.File(file_path, 'r') as f:
        print(f.keys())
        # Define groups of keys for better organization
        groups = {
            "Metadata": ['#refs#', '#subsystem#', 'annotatedOnly', 'curatedOn', 'detectedOn', 'sortTime', 'sortedOn'],
            "Cluster Information": ['clusterCenters', 'clusterCentroids', 'clusterNotes', 'clusterSites', 'unitCount', 
                                     'unitISIRatio', 'unitIsoDist', 'unitLRatio', 'unitPeakSites', 'unitPeaks', 
                                     'unitPeaksRaw', 'unitSNR', 'unitVpp', 'unitVppRaw', 'waveformSim'],
            "Spike Information": ['spikeAmps', 'spikeClusters', 'spikeDelta', 'spikeNeigh', 'spikePositions', 
                                   'spikeRho', 'spikeSites', 'spikeSites2', 'spikeSites3', 'spikeTimes', 
                                   'spikesByCluster', 'spikesBySite', 'spikesBySite2', 'spikesBySite3'],
            "Waveform Information": ['meanWfGlobal', 'meanWfGlobalRaw', 'meanWfLocal', 'meanWfLocalRaw', 
                                      'meanWfRawHigh', 'meanWfRawLow', 'featuresShape', 'filtShape', 'rawShape'],
            "Thresholds and Metrics": ['meanSiteThresh', 'nSitesOverThresh', 'siteRMS', 'siteThresh', 
                                        'rhoCutGlobal', 'rhoCutSite', 'rhoCuts', 'ordRho'],
            "History": ['history', 'editPos', 'detectTime']
        }

        # Print keys for each group
        for group_name, keys in groups.items():
            print(f"\n{'=' * 40}")
            print(f"Group: {group_name}")
            print(f"{'=' * 40}")
            for key in keys:
                try:
                    if key in f and key not in ['spikesBySite', 'spikesBySite2', 'spikesBySite3']:
                        data = f[key]
                        # Check if the dataset is small enough to print
                        if data.size <= max_elements:
                            print(f"{key}: {data.shape} {data.dtype} -> {data[()]}")
                        else:
                            print(f"{key}: {data.shape} {data.dtype} (Too large to print)")
                    else:
                        print(f"{key}: Key not found")
                except Exception as e:
                    print(f"{key}: Failed to unpack ({e})")

        # Print any remaining keys not in predefined groups
        remaining_keys = set(f.keys()) - {key for group in groups.values() for key in group}
        if remaining_keys:
            print(f"\n{'=' * 40}")
            print("Group: Other Keys")
            print(f"{'=' * 40}")
            for key in remaining_keys:
                try:
                    data = f[key]
                    # Check if the dataset is small enough to print
                    if data.size <= max_elements:
                        print(f"{key}: {data.shape} {data.dtype} -> {data[()]}")
                    else:
                        print(f"{key}: {data.shape} {data.dtype} (Too large to print)")
                except Exception as e:
                    print(f"{key}: Failed to unpack ({e})")
