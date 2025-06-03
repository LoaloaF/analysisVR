import os
import time

import numpy as np
import pandas as pd

import h5py

from CustomLogger import CustomLogger as Logger
import analytics_processing.sessions_from_nas_parsing as sp
import analytics_processing.modality_loading as ml
from analytics_processing.analytics_constants import device_paths

# from scipy.signal import butter
# from scipy.signal import filtfilt

# slow imports requing C compilation
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

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

    for i, (session_fullfname, identif) in enumerate(zip(sessionlist_fullfnames, ids)):
        L.logger.info(f"Processing {identif} ({i+1}/{len(ids)}) "
                      f"{os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        
        # create the dat file 
        _handle_session_ephys(session_fullfname, mode, exclude_shanks)
        
def _unpack_JRC_output(session_fullfname, get_spikes=False, 
                       get_cluster_metadata=False, get_spikes_cluster_subset=None):
    L = Logger()
    
    def ssbatch_cluster_metadata():
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
                "cluster_channel": f['clusterSites'][:, 0]-1,
                "cluster_id_ssbatch": clust_id,
                "unit_snr": f['unitSNR'][0],
                "unit_Vpp": f['unitVpp'][0] /6.3, # error in JRC settings, reset here
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
        L.logger.debug(f"Spike cluster table:\n{cluster_table}")
        return cluster_table
        
    def ssbatch_aggr_spikes():
        with h5py.File(ss_res_fullfname, 'r') as f:
            ss_spike_times = f['spikeTimes'][0]
            ss_from_spike_i = np.where((session_from_smple < ss_spike_times))[0][0]
            ss_to_spike_i = np.where((session_to_smple < ss_spike_times))[0]
            if len(ss_to_spike_i) != 0:
                ss_to_spike_i = ss_to_spike_i[0]
            else:
                # last spike before end of session
                ss_to_spike_i = len(ss_spike_times)-1
            L.logger.debug(f"Session samples within concatenated data conext go"
                        f" from {session_from_smple:,} to {session_to_smple:,}." 
                        f"All spike ({len(ss_spike_times):,}) go from sample {ss_spike_times[0]:,} to {ss_spike_times[-1]:,}."
                        f" This session's spikes are at: [{ss_spike_times[ss_from_spike_i]:,} ... {ss_spike_times[ss_to_spike_i]:,}]")
            
            spike_times = ss_spike_times[ss_from_spike_i:ss_to_spike_i]
            spike_clusters = f['spikeClusters'][0, ss_from_spike_i:ss_to_spike_i]
            spike_amps = f['spikeAmps'][0, ss_from_spike_i:ss_to_spike_i]
            spike_positions = f['spikePositions'][0, ss_from_spike_i:ss_to_spike_i]
            spike_sites = f['spikeSites'][0, ss_from_spike_i:ss_to_spike_i]
            spike_sites_2nd = f['spikeSites2'][0, ss_from_spike_i:ss_to_spike_i]
            if f['spikeSites3'].shape == (2,): # JRC output might not have 3rd site
                spike_sites_3rd = np.array([np.nan]*len(spike_sites_2nd))
                chnls_str = ['channel', 'channel_2nd']
                nsites = 2
            else: 
                spike_sites_3rd = f['spikeSites3'][0, ss_from_spike_i:ss_to_spike_i]
                chnls_str = ['channel', 'channel_2nd', 'channel_3rd']
                nsites = 3
        
        spike_table = pd.DataFrame({
            "sample_id": spike_times-session_from_smple,
            "ephys_timestamp": (spike_times-session_from_smple).astype(np.uint64)*50,
            "cluster_id_ssbatch": spike_clusters.astype(int),
            "channel": spike_sites-1,
            "amplitude_uV": spike_amps,
            "ss_batch_id": ss_i,
            "channel_hpf_wf": pd.NA,
            "channel_2nd_hpf_wf": pd.NA,
            "channel_3rd_hpf_wf": pd.NA,
            "channel_2nd": spike_sites_2nd-1,
            "channel_3rd": spike_sites_3rd-1,
            "shank": np.round(spike_positions)//1000,
        })
        # remove hard deleted clusters, using backspace in JRC (-1)
        spike_table = spike_table.loc[spike_table.cluster_id_ssbatch!=-1].reset_index(drop=True)
        if get_spikes_cluster_subset is not None:
            incl_clust = get_spikes_cluster_subset[get_spikes_cluster_subset.ss_batch_id == ss_i].cluster_id_ssbatch
            spike_table = spike_table[spike_table.cluster_id_ssbatch.isin(incl_clust)]
        
        # drop ISI violations, targeting case: same cluster, different site, ISI < 1ms
        def drop_isi_viol(cluster_spikes):
            isi = np.diff(cluster_spikes.ephys_timestamp.values)
            return cluster_spikes.iloc[np.where(isi > 1_000)[0]+1]
        prv_nspikes = spike_table.shape[0]
        spike_table = spike_table.groupby('cluster_id_ssbatch').apply(drop_isi_viol)
        
        # add depth
        spike_table['depth'] = mapping.depth.values[spike_table.channel]
        spike_table = spike_table.sort_values(by=['sample_id']).reset_index(drop=True)
        L.logger.debug(f"Extracted spikes from {spike_table.cluster_id_ssbatch.nunique():,} "
                       f" unique clusters, dropped {prv_nspikes - spike_table.shape[0]:,} "
                       f" spikes for ISI violations.\n{spike_table}")
        
        def process_spike_chunk(spike_table_chunk, sos_filter):
            # create the time window around the spike chunk
            from_smple, to_smple = spike_table_chunk.iloc[[0,-1]].sample_id.values
            from_smple = max(from_smple - filt_window//2, 0)
            to_smple = min(to_smple + filt_window//2, traces.shape[1])
            # realign spike sample_ids to the chunk, starting from 0
            spike_table_chunk.loc[:, 'sample_id'] -= from_smple
            # read raw data to memory
            traces_chunk = np.array(traces[:, from_smple:to_smple], dtype=np.int16)
            
            # create the channel index, one flat list is the final output
            rows_2d = spike_table_chunk.loc[:, chnls_str].values
            rows_flat = np.repeat(rows_2d.flatten(), filt_window)
            
            # create the tinewindows, repeat index for filtwindow & site to make flat
            t = np.repeat(spike_table_chunk.sample_id.values, nsites) # peaks, nsites
            cols_2d = np.row_stack([np.arange(-filt_window//2, filt_window//2)]*len(t)) +t[:, None] # expand over window
            cols_2d = np.clip(cols_2d, 0, traces_chunk.shape[1]-1) # ensure within sample bounds
            cols_flat = cols_2d.flatten()
            
            waveforms = traces_chunk[rows_flat, cols_flat].reshape(nsites, spike_table_chunk.shape[0], filt_window)
            t2 = time.time()

            # iterate over the sites and filter the waveforms
            all_filt_wfs = [[] for _ in range(nsites)]
            for sp_i in range(waveforms.shape[1]):
                for site_k in range(nsites):
                    filt_wf = np.round(filtfilt(*sos_filter, waveforms[site_k, sp_i, :])).astype(np.int16)
                    all_filt_wfs[site_k].append(filt_wf[filt_window//2 -10:-filt_window//2 +15])
            # print(f"Filtered waveforms took {time.time()-t2:.3f} s")
            
            # instert spike waveform into spike_table
            all_filt_wfs_aggr = []
            for site_k, col_name in enumerate(chnls_str):
                all_filt_wfs_aggr.append(pd.Series(all_filt_wfs[site_k], 
                                                    name=col_name+'_hpf_wf',))

            return pd.concat(all_filt_wfs_aggr, axis=1)
        
        # # skip this if don't want to wait 20 minutes per sessions
        # # TODO parallelize this once server reads are concurrent
        # L.logger.debug(f"Extracting wavefroms & filtering...")
        # nspikes_per_read = 10
        # filt_window = 512
        # sos_filter = butter(4, [300, 5000], btype='band', fs=20_000)
        # start_t = time.time()
        # # for chunk_i in range(spike_table.shape[0] //nspikes_per_read):
        # #     from_i, to_i = chunk_i*nspikes_per_read, (chunk_i+1)*nspikes_per_read
        # for from_i in spike_table.index[::nspikes_per_read]:
        #     to_i = min(from_i + nspikes_per_read, spike_table.shape[0])
        #     waveforms = process_spike_chunk(spike_table.loc[from_i:to_i].copy(),
        #                                     sos_filter)
        #     # add the waveforms to the spike table
        #     spike_table.loc[from_i:to_i, waveforms.columns] = waveforms.values
            
        #     sp_per_second = to_i /(time.time()-start_t)
        #     print(f"{to_i:07_d}/{spike_table.shape[0]:_} spikes, "
        #             f"{(spike_table.shape[0]-to_i) /sp_per_second /60:.2f} "
        #             f"minutes left at {sp_per_second:.2f} spikes/s", 
        #             end='\r' if to_i < spike_table.shape[0] else 
        #             f'\ntook {(time.time()-start_t)/60:.2f}min')
        return spike_table

    # ==========================================================================
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
    # print(f"Mapping:\n{mapping}")   

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
            # get the cluster metadata
            cluster_table = ssbatch_cluster_metadata()
            aggr_over_ss_batches.append(cluster_table)
            
        elif get_spikes:
            # get the spikes
            spike_table = ssbatch_aggr_spikes()
            aggr_over_ss_batches.append(spike_table)
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
    spikes = _unpack_JRC_output(session_fullfname, get_spikes=True,
                                get_spikes_cluster_subset=sp_clust_metadata.loc[:, ['ss_batch_id','cluster_id_ssbatch']])
    # check if the last step is far away from the last sample
    last_spike_gap = sp_clust_metadata.session_nsamples.iloc[0] - spikes.sample_id.values[-1]
    if last_spike_gap > 100: # 5 ms
        Logger().logger.warning(f"End of session has {last_spike_gap /20_000:.3f}"
                                f"s with no spikes!")
    
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
    print(spikes['ephys_timestamp'])
    breaks = np.arange(0, spikes['ephys_timestamp'].max()+bin_size_us, 
                       bin_size_us, dtype='uint64')
    spikes['bin_200ms'] = pd.cut(
        spikes['ephys_timestamp'],
        bins=pd.IntervalIndex.from_breaks(breaks)
    )

    # then group by clusterID and use the bin_200ms to count spikes
    cnt = lambda x: x['bin_200ms'].value_counts()/0.2
    fr_40ms_hz = spikes.groupby('cluster_id').apply(cnt, include_groups=False)
    fr_40ms_hz = fr_40ms_hz.unstack().fillna(0).T
    
    # append the bin edges    
    fr_40ms_hz['from_ephys_timestamp'] = breaks[:-1]
    fr_40ms_hz['to_ephys_timestamp'] = breaks[1:]
    # return the riting rate with time_bins on the index 0,1,.. and cluster_ids 
    # +bin edges on the columns
    return fr_40ms_hz.reset_index(drop=True)

def get_FiringRate40msZ(fr_hz):
    def zscore(x):
        return (x - x.mean()) / x.std()
    return fr_hz.apply(zscore, axis=0)

# def get_FiringRateTrackbinsHz(fr_z, track_data):
    
#     def trackwise_averages(fr, track_data):
#         def time_bin_avg(trial_data, ):
#             print(f"{trial_data.from_z_position_bin.iloc[0]:04}", end='\r')
#             trial_aggr = []
#             for row in range(trial_data.shape[0]):
#                 from_t, to_t = trial_data.iloc[row].loc[['posbin_from_ephys_timestamp', 'posbin_to_ephys_timestamp']]
#                 from_t = from_t - 40_000
#                 to_t = to_t + 40_000
                
#                 trial_d_track_bin_fr = fr.loc[(fr.index.left > from_t) & (fr.index.right < to_t)]
#                 # print(trial_data.iloc[row])
#                 # print(trial_d_track_bin_fr.shape, to_t-from_t, )
#                 # print()
#                 trial_aggr.append(trial_d_track_bin_fr.values)
#             trial_aggr_concat = np.concatenate(trial_aggr, axis=0)
#             if trial_aggr_concat.shape[0] == 0:
#                 # no trials in this bin
#                 return pd.Series(np.nan, index=fr.columns)
#             m = trial_aggr_concat.mean(axis=0)
#             return pd.Series(m, index=fr.columns)
                
#         print("Track bin: ")
#         return track_data.groupby('from_z_position_bin', ).apply(time_bin_avg)
    
#     # recover index
#     old_idx = fr_z.index
#     fr_z = fr_z.reset_index(drop=True)
#     new_idx = pd.IntervalIndex.from_breaks(
#         np.arange(0, fr_z.shape[0]*40_000+1, 40_000, dtype='uint64'),
#     )
#     fr_z.index = new_idx
#     fr_hz_averages = trackwise_averages(fr_z, track_data)
#     return fr_hz_averages
#     import matplotlib.pyplot as plt
#     plt.imshow(fr_hz_averages.values.T, aspect='auto', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#     print(fr_hz_averages)

def get_FiringRateTrackwiseHz(fr, track_data):
    def time_bin_avg(posbin_data, ):
        # cue = posbin_data.cue.iloc[0]
        # trial_outcome = posbin_data.trial_outcome.iloc[0]
        pos_bin = posbin_data.from_position_bin.iloc[0]
        print(f"{pos_bin:04}", end='\r')
        trial_aggr = []
        # print("--------------------")
        # print(f"{pos_bin=}, trial_id:")
        # print(posbin_data.trial_id)
        # print(posbin_data)
        
        
        # for tr_id in posbin_data.trial_id:
        #     trial_data = posbin_data.loc[posbin_data.trial_id==tr_id]
        #     from_t, to_t = trial_data.loc[:, ['posbin_from_ephys_timestamp', 'posbin_to_ephys_timestamp']].values[0]
        #     from_t = from_t - 40_000
        #     to_t = to_t + 40_000
            
        #     # print(from_t, to_t)
        #     # print(fr)
        #     trial_d_track_bin_fr = fr.loc[(fr.index.left > from_t) & (fr.index.right < to_t)]
        #     m = trial_d_track_bin_fr.mean(axis=0)
        #     trial_aggr.append(pd.Series(m, index=fr.columns, name=(pos_bin,tr_id,cue,trial_outcome)))
        
        
        # to_posbin_t = trial_data.posbin_from_ephys_timestamp
        from_posbin_t = posbin_data.posbin_from_ephys_timestamp.values
        to_posbin_t = posbin_data.posbin_to_ephys_timestamp.values
        interval = pd.IntervalIndex.from_arrays(
            from_posbin_t-40_000,
            to_posbin_t+40_000,
            closed='both',
        )
        # print(interval)
        # print(fr.index.mid)
        assigned_bin = pd.cut(fr.index.mid,
                              bins=interval,
                              labels=posbin_data.trial_id[:-1],
        )
        trials_exist_mask = (assigned_bin.value_counts() != 0).values
        # print(f"Trials exist mask sum: {trials_exist_mask.sum()}")
        
        # slice out the 40ms fr bins that overlap with the trial-wise position bin
        trial_fr = fr[assigned_bin.notna()].copy().astype(np.float32)
        # add a column indicating the 
        trial_fr['posbin_t_edges'] = assigned_bin[assigned_bin.notna()]
        # print(trial_fr.shape)
        posbin_trial_wise_fr = trial_fr.groupby('posbin_t_edges', observed=True).mean()
        # print(posbin_trial_wise_fr.shape)
        posbin_trial_wise_fr /= interval.length.values[trials_exist_mask, None] /1e6 # us to s
        # print(posbin_trial_wise_fr.shape)

        # add meta data, cue outcome, position bin, trial id        
        posbin_trial_wise_fr['cue'] = posbin_data[trials_exist_mask].cue.values
        posbin_trial_wise_fr['trial_outcome'] = posbin_data[trials_exist_mask].trial_outcome.values
        posbin_trial_wise_fr['choice_R1'] = posbin_data[trials_exist_mask].choice_R1.values
        posbin_trial_wise_fr['choice_R2'] = posbin_data[trials_exist_mask].choice_R2.values
        posbin_trial_wise_fr['bin_length'] = interval.length.values[trials_exist_mask]/1e6
        posbin_trial_wise_fr.index = posbin_data.trial_id.values[trials_exist_mask]
        # print(posbin_trial_wise_fr)
        # print("---")
        # exit()
        return posbin_trial_wise_fr        

    # recover interval index
    fr.index = pd.IntervalIndex.from_arrays(fr.pop("from_ephys_timestamp"), 
                                            fr.pop("to_ephys_timestamp"))

    print("Track bin: ")
    fr_hz_averages = track_data.groupby(['from_position_bin']).apply(time_bin_avg)
    fr_hz_averages.index = fr_hz_averages.index.rename(['from_position_bin', 'trial_id'],)
    fr_hz_averages.reset_index(inplace=True, drop=False)
    # print(fr_hz_averages)
    # exit()
    return fr_hz_averages

def get_PCsZonewise(fr, track_data):
    L = Logger()
    PCs_var_explained = .8
    fr = fr.set_index(['trial_id', 'from_position_bin', 'cue', 'trial_outcome', 
                       'choice_R1', 'choice_R2'], append=True, )
    fr.index = fr.index.droplevel(3) # entry_id

    fr.drop(columns=['bin_length', ], inplace=True)
    fr.columns = fr.columns.astype(int)
    fr = fr.reindex(columns=sorted(fr.columns))
    
    track_data = track_data.set_index(['trial_id', 'from_position_bin', 'cue', 'trial_outcome', 
                                       'choice_R1', 'choice_R2'], append=True, )
    track_data.index = track_data.index.droplevel(3) # entry_id
    
    zones = {
        'beforeCueZone': (-168, -100),
        'cueZone': (-80, 25),
        'afterCueZone': (25, 50),
        'reward1Zone': (50, 110),
        'betweenRewardsZone': (110, 170),
        'reward2Zone': (170, 230),
        'postRewardZone': (230, 260),
        'wholeTrack': (-168, 260),
    }
    
    aggr_embeds = []
    aggr_subspace_basis = []

    #TODO each sessions has differnt number of neurons
    fr = fr.reindex(np.arange(1,78), axis=1).fillna(0)

    for i in range(4):
        # which modality to use for prediction
        if i == 0:
            predictor = fr.iloc[:, :20]
            predictor_name = 'HP'
        elif i == 1:   
            predictor = fr.iloc[:, 20:]
            predictor_name = 'mPFC'
        elif i == 2:
            predictor = track_data.loc[:, ['posbin_velocity', 'posbin_acceleration', 'lick_count',
                                           'posbin_raw', 'posbin_yaw', 'posbin_pitch']]
            predictor_name = 'behavior'
        elif i == 3:
            predictor = fr.iloc[:]
            predictor_name = 'HP-mPFC'
            
        debug_msg = (f"Embedding {predictor_name} with PCA:\n")
        for zone, (from_z, to_z) in zones.items():
            zone_data = predictor.loc[pd.IndexSlice[:,:,:,:,np.arange(from_z,to_z)]].astype(float)
            
            # Standardize features
            scaler = StandardScaler()
            zone_data.loc[:,:] = scaler.fit_transform(zone_data)
             
            zone_data_Z_trialwise = zone_data.unstack(level='from_position_bin')
            zone_data_Z_trialwise.index = zone_data_Z_trialwise.index.droplevel([0,1,2])
            zone_data_Z_trialwise = zone_data_Z_trialwise.fillna(0).astype(float)
            
            pca = PCA(n_components=PCs_var_explained)
            embedded = pca.fit_transform(zone_data_Z_trialwise)
            
            debug_msg += (f"{embedded.shape[1]:3d} PCs capture {pca.explained_variance_ratio_.sum():.2f} "
                          f" of variance of {zone_data_Z_trialwise.shape[1]:5d} "
                          f"input dims in {embedded.shape[0]:3d} trials at "
                          f"{zone} ({from_z}, {to_z})\n")

            embedded = np.round(embedded, 3)
            aggr_embeds.append(pd.Series(embedded.tolist(), index=zone_data_Z_trialwise.index,
                                         name=(zone, predictor_name)))
            
            columns = pd.MultiIndex.from_product([[zone], [predictor_name], 
                                                  np.arange(embedded.shape[1])])
            aggr_subspace_basis.append(pd.DataFrame(pca.components_.round(3).T,
                                                    columns=columns))
        L.logger.debug(debug_msg)

    aggr_embeds = pd.concat(aggr_embeds, axis=1).stack(level=0, future_stack=True)
    aggr_embeds.reset_index(inplace=True, drop=False)
    aggr_embeds.rename(columns={"level_5": "track_zone"}, inplace=True)
    
    # the subspace basis is a 2D array with pricipal components as columns, rows
    # indicate predictors and track zones
    #TODO warning right now due to old stacking behavior, will be fixed in future
    aggr_subspace_basis = pd.concat(aggr_subspace_basis, axis=0).stack(level=(0,1), future_stack=False)
    aggr_subspace_basis.reset_index(inplace=True, drop=False)
    aggr_subspace_basis.rename(columns={"level_1": "track_zone",
                                        "level_2": "predictor_name",}, inplace=True)
    aggr_subspace_basis.drop(columns='level_0', inplace=True)
    
    # check for in proper sessions that lack spikes for most of the session
    valid_ephys_mask = aggr_embeds['mPFC'].notna()
    valid_beh_mask = aggr_embeds['behavior'].notna()
    if not all(valid_ephys_mask == valid_beh_mask):
        msg = (f"Ephys and behavior trials mismatch!\n"
               f"Valid trials in ephys: {aggr_embeds.trial_id[valid_ephys_mask].unique()}, "
               f"valid trials in behavior: {aggr_embeds.trial_id[valid_beh_mask].unique()} "
               f"Using common set of trials")
        Logger().logger.warning(msg)
        aggr_embeds.dropna(how='any', inplace=True)
    
    return aggr_subspace_basis, aggr_embeds

def get_PCsSubspaceAngles(session_subspace_basis, all_subspace_basis):
    session_subspace_basis.index = session_subspace_basis.index.droplevel((0,1))  # Drop the first level of the index if it exists
    session_subspace_basis.set_index(["track_zone", 'predictor_name'], inplace=True, append=True)
    session_subspace_basis.index = session_subspace_basis.index.droplevel("entry_id")  # Drop the entry_id level if it exists
    
    all_subspace_basis.index = all_subspace_basis.index.droplevel((0,1))  # Drop the first level of the index if it exists
    all_subspace_basis.set_index(["track_zone", 'predictor_name'], inplace=True, append=True)
    all_subspace_basis.index = all_subspace_basis.index.droplevel("entry_id")  # Drop the entry_id level if it exists
    # print(all_subspace_basis)
    
    angle_aggr = []
    comps_aggr = []
    for zone in all_subspace_basis.index.unique(level='track_zone'):
        print(zone, end='\r')
        for predictor in all_subspace_basis.index.unique(level='predictor_name'):
            # get the subspace basis for the session
            # track_session_subspace = session_subspace_basis.loc[(zone, predictor), :]
            # get the subspace basis for all sessions
            zone_all_subspace = all_subspace_basis.loc[(slice(None),zone, predictor), :]
            session_n_PCs = zone_all_subspace.groupby(level='session_id').apply(
                                                       lambda x: x.notna().all(0).sum())
            # some sessions have very low number of PCs, exclude those, will be NaN in the end
            std = np.std(session_n_PCs.values)
            cutoff = np.mean(session_n_PCs.values) - 2*std
            
            too_few_PCs_mask = session_n_PCs < cutoff
            n_PCs = session_n_PCs[~too_few_PCs_mask].min()
            
            # # print(track_all_subspace)
            # print(zone, predictor,)
            # smth = session_n_PCs.to_frame(name='n_PCs').copy()
            # smth['keep'] = smth.n_PCs > cutoff
            # print(smth)
            # print('-----')
            
            # print(zone, predictor,)
            # print(n_PCs)
            # n_PCs = 200
            
            # zone_all_subspace = zone_all_subspace.iloc[:, :n_PCs]
            # zone_session_subspace = , :].iloc[:, :n_PCs]
            
            # print(zone_all_subspace, zone_session_subspace)
            s0_subspace = session_subspace_basis.loc[(slice(None), zone, predictor)].values[:, :n_PCs]
            if np.isnan(s0_subspace[:, :n_PCs]).any() or s0_subspace.shape[1] < n_PCs:
                Logger().logger.warning(f"Session subspace has too few PCs for {zone} {predictor}.")
                continue
            
            for s_id in session_n_PCs[~too_few_PCs_mask].index:
                s_compare_subspace = zone_all_subspace.loc[s_id, :].values[:, :n_PCs]
                # print(zone_all_subspace.loc[s_id, :])
                
                # print(s0_subspace, s_compare_subspace)
                # print(s0_subspace.shape, s_compare_subspace.shape)
                
                # print("--")
                M = s0_subspace.T @ s_compare_subspace
                s0_c, S, s_comp_h_c = np.linalg.svd(M)
                

                canonc_angles = np.arccos(np.clip(S, -1, 1))
                s0_c_subspace = (s0_subspace @ s0_c).astype(np.float16)
                # print(s0_c_subspace)
                # print(s0_c_subspace.shape)
                # CC_U = (s1_PCs_U @ U_c[:, :])
                # print(c1_U)
                # print(c1_U.shape)
                
                # canonc_angles = subspace_angles(s0_subspace[:, :n_PCs],
                #                                 s_compare_subspace[:, :n_PCs])
                # canonc_angles = np.rad2deg(canonc_angles)
                # print(canonc_angles)
                
                angle_aggr.append(pd.Series(canonc_angles, index=np.arange(n_PCs),
                                            name=(zone, predictor, s_id)))
                comps_aggr.append(pd.DataFrame(s0_c_subspace, columns=pd.MultiIndex.from_product(
                                                 [[zone], [predictor], [s_id], np.arange(n_PCs)],
                                                 names=['track_zone', 'predictor', 'comp_session_id', 'CC_i'],
                                             )))
                
                
                print(comps_aggr[-1])
    if len(angle_aggr) == 0:
        return None
    
    angle_aggr = pd.concat(angle_aggr, axis=1).T
    angle_aggr.index.names = ['track_zone', 'predictor', 'comp_session_id']
    
    comps_aggr = pd.concat(comps_aggr, axis=1)
    comps_aggr = comps_aggr.T
    return comps_aggr.reset_index(), angle_aggr.reset_index()
            
def get_SVMCueOutcomeChoicePred(PCsZonewise):
    L = Logger()

    def fit_SVMs_with_bootstrap(X, Ys, name, n_iterations=200):
        predictions = []
        rng = np.random.default_rng(42)

        for Y_name, y in Ys.items():
            # if Y_name != 'choice_R1':
            #     continue
            lbl_counts = pd.Series(y).value_counts()
            print(Y_name)
            if lbl_counts.min() < 5 or len(lbl_counts) < 2:
                L.logger.warning(f"Not enough samples for {Y_name} in {name}: ({lbl_counts})")
                continue
            elif len(lbl_counts) != 2:
                L.logger.warning(f"More than 2 classes for {Y_name} in {name}: ({lbl_counts})")
                continue

            for kernel in ('linear', 'rbf'):
                # if kernel != 'linear':
                #     continue
                Cs = [0.1, .5, 1, 5, 10,]
                accs, f1s = [], []

                aggr_predictions = np.ones((n_iterations, len(X)), dtype=int) * -1
                print(aggr_predictions.shape)
                for i in range(n_iterations):
                    indices = rng.choice(len(X), size=len(X), replace=True)
                    X_boot, y_boot = X[indices], y[indices]
                    # only a single cclass check:
                    if len(np.unique(y_boot)) < 2:
                        print("Skipping SVM fit due to single class in bootstrap sample")
                        continue
                    oob_mask = np.ones(len(X), dtype=bool)
                    oob_mask[indices] = False
                    X_oob, y_oob = X[oob_mask], y[oob_mask]

                    if len(y_oob) < 5 or len(np.unique(y_oob)) < 2:
                        continue

                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svc', SVC(kernel=kernel, gamma='scale'))
                    ])
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid={'svc__C': Cs},
                        cv=6,
                        scoring='balanced_accuracy',
                        n_jobs=-1,
                        verbose=False,
                    )
                    grid_search.fit(X_boot, y_boot)
                    y_pred = grid_search.predict(X_oob)
                    report = classification_report(y_oob, y_pred, output_dict=True, zero_division=0)

                    accs.append(balanced_accuracy_score(y_oob, y_pred))
                    f1s.append(report['macro avg']['f1-score'])
                    
                    # every row is one bootstrap iteration
                    aggr_predictions[i, oob_mask] = y_pred
                
                # get the "average" prediction across bootstrap iterations
                aggr_predictions = pd.DataFrame(aggr_predictions)
                aggr_predictions[aggr_predictions == -1] = np.nan
                pred = aggr_predictions.mode(axis=0).iloc[0].values
                
                # difference
                f1_aggr = classification_report(y, pred, output_dict=True, zero_division=0)['macro avg']['f1-score']
                
                print(f"mean: {np.mean(f1s):.3f}, aggr: {f1_aggr:.3f}, "
                      f"diff: {np.mean(f1s) - f1_aggr:.3f}")
                
                # # Fit final model on full data for predictions
                # final_model = GridSearchCV(
                #     estimator=Pipeline([
                #         ('scaler', StandardScaler()),
                #         ('svc', SVC(kernel=kernel, gamma='scale'))
                #     ]),
                #     param_grid={'svc__C': Cs},
                #     cv=6,
                #     scoring='f1_macro',
                #     n_jobs=-1
                # )
                # final_model.fit(X, y)
                # pred = final_model.predict(X)
                # # print(np.stack((y, pred)).T)
                
                # print(f"---\n{Y_name} {name} {kernel}")
                # if Y_name == 'cue' and kernel == 'linear' and name[1] == 'afterCueZone' and name[0] == 'HP':
                #     print(f"---\n{Y_name} {name} {kernel}")
                #     print(classification_report(y_oob, y_pred))
                #     print(pred)
                #     print(y)
                #     # exit()
                

                col_index = pd.MultiIndex.from_tuples([(*list(name), kernel, Y_name, n) for n in 
                                                    ('y', 'y_true', 'n_PCs', 'acc', 'acc_std', 'f1')],
                                                    names=['predictor', 'track_zone', 'model', 'predicting', 'output'])
                pred_output = np.concatenate([
                    pred[:, None], y[:, None],
                    np.tile(np.array([X.shape[1], np.mean(accs), np.std(accs), np.mean(f1s)]), (len(pred), 1))
                ], axis=1)

                predictions.append(pd.DataFrame(pred_output, columns=col_index, index=PCsZonewise.index))
        if predictions == []:
            L.logger.warning(f"None enough trials to fit any SVM for {name}")
            return None
        return pd.concat(predictions, axis=1)
    


    
    
    

    # fix seed
    np.random.seed(42)

    def parse_string_to_array(x):
        # TODO why?
        if isinstance(x, np.ndarray):
            return x
        if pd.isna(x):
            return x
        return np.fromstring(x.strip("[]"), sep=", ", dtype=np.float32)
    
    PCsZonewise = PCsZonewise.set_index(['track_zone', 'trial_id', 'trial_outcome', 
                                         'cue', 'choice_R1', 'choice_R2'], 
                                        append=False).unstack(level='track_zone')
    all_zones = []
    for column in PCsZonewise.columns:
        # if column[1] != 'beforeCueZone':
        #     continue
        # if column[0] in ('HP', 'mPFC'):
        #     continue
        X_list = PCsZonewise[column].apply(parse_string_to_array)
        if X_list.isna().any():
            L.logger.warning(f"Missing trials for {column}...")
            mask = X_list.isna()
        else:
            mask = np.ones(X_list.shape, dtype=bool)
        X = np.stack(X_list[mask].values)
            
        if X.shape[0] <8:
            Logger().logger.warning(f"Minimum of 8 trials req. to fit SVM. "
                                    f"{X.shape[0]} trials found for {column}")
            continue
        Logger().logger.debug(f"Fitting SVM with {column}...")
        # print(PCsZonewise)
        Y_cue = PCsZonewise[mask].index.get_level_values('cue').values
        Y_outcome = PCsZonewise[mask].index.get_level_values('trial_outcome').values
        Y_outcome = Y_outcome.astype(bool).astype(int)
        # TODO: add choice
        Y_choice_R1 = PCsZonewise[mask].index.get_level_values('choice_R1').values.astype(int)
        Y_choice_R2 = PCsZonewise[mask].index.get_level_values('choice_R2').values.astype(int)
        
        predictions = fit_SVMs_with_bootstrap(X, Ys={'cue':Y_cue, 'outcome':Y_outcome, 'choice_R1':Y_choice_R1, 'choice_R2': Y_choice_R2}, name=column)
        all_zones.append(predictions)

    if all_zones == [] or all(pr is None for pr in all_zones):
        L.logger.warning("None of zones in the session have min. trials == 6")
        return None
    
    all_zones = pd.concat(all_zones, axis=1)
    all_zones = all_zones.stack(level=('predictor', 'track_zone', 'model', 'predicting'), future_stack=True).reset_index()
    # print(all_zones.columns)
    # print(all_zones)
    return all_zones
