import os
import time
import json

import numpy as np
import pandas as pd
import re

import h5py

from CustomLogger import CustomLogger as Logger
import analytics_processing.sessions_from_nas_parsing as sp
import analytics_processing.modality_loading as ml
from analytics_processing.analytics_constants import device_paths
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy import stats

# slow imports requing C compilation
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from dashsrc.plot_components.plot_utils import make_discr_cluster_id_cmap

# from ../../ephysVR.git

try:
    from mea1k_modules.mea1k_post_processing import mea1k_raw2decompressed_dat_file
except ImportError:
    Logger().logger.error("Could not import mea1k_modules.mea1k_post_processing."
                          " Make sure ephysVR repository is available.")

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
        session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
        session_dir = os.path.dirname(session_fullfname)
        animal_name = session_name.split("_")[2]
        
        # check if a dat file already exists
        dat_fnames = [f for f in os.listdir(session_dir) if f.endswith("ephys_traces.dat")]
        dat_sizes = np.array([os.path.getsize(os.path.join(session_dir, f)) for f in dat_fnames])

        if any(dat_sizes>0):
            L.logger.debug(f"De-compressed ephys files found: {dat_fnames}")
            if mode != "recompute":
                L.logger.debug(f"Skipping...")
                continue
            else:
                L.logger.debug(f"Recomputing...")
            
        if os.path.exists(os.path.join(session_dir, "ephys_output.raw.h5")):
            fname = 'ephys_output.raw.h5'
        elif os.path.exists(os.path.join(session_dir, f"{session_name}_ephys_logger.raw.h5")):
            fname = f"{session_name}_ephys_logger.raw.h5"
        else:
            L.logger.warning(f"No raw ephys recordings found for {session_name}")
            continue

        # decompress the raw output of mea1k and convert to uV int16 .dat file
        # also, save a mapping csv file for identifying the rows in the dat file
        mea1k_raw2decompressed_dat_file(session_dir, 
                                        fname=fname, 
                                        session_name=session_name,
                                        animal_name=animal_name,
                                        convert2uV=True,
                                        subtract_dc_offset=True,
                                        write_neuroscope_xml=True,
                                        write_probe_file=True,
                                        replace_with_curated_xml=False,
                                        exclude_shanks=exclude_shanks)
        
        
        
        
        
        
        
        
        
        
def extract_jrc_spikes(session_fullfname): #, get_spikes=False, 
                                           #    get_cluster_metadata=False, get_spikes_cluster_subset=None):
    L = Logger()
    
    def ssbatch_cluster_metadata(ss_res_fullfname):
        # open jrc output to read cluster /specifc unit metadata, excluding single spikes
        with h5py.File(ss_res_fullfname, 'r') as f:
            cluster_notes_data = []
            # unpack jrc's messy notes
            for note_raw in [f[obj_ref][()] for obj_ref in f['clusterNotes'][0]]:
                cluster_notes_data.append("".join([chr(c.item()) for c in note_raw]))
            cluster_notes_data = [n if n != '\x00\x00' else '' for n in cluster_notes_data]
            clust_id = np.sort(np.unique(f['spikeClusters'][0]))
            clust_id = clust_id[~np.isin(clust_id, (-1, 0))].astype(int)
            cluster_table = pd.DataFrame({
                "cluster_id": clust_id +(10_000*ss_i), # make ids unique
                "cluster_type": cluster_notes_data,
                "unit_count": f['unitCount'][0],
                "cluster_channel": f['clusterSites'][:, 0]-1, # matlab 1-based index
                "cluster_id_ssbatch": clust_id,
                "unit_snr": f['unitSNR'][0],
                "unit_Vpp": f['unitVpp'][0] /6.3, # error in JRC settings, reset here, TODO rm later
                "unit_isi_ratio": f['unitISIRatio'][:,0],
                "unit_iso_dist": f['unitIsoDist'][:,0],
                "unit_L_ratio": f['unitLRatio'][:,0],
            })
        cluster_table['session_nsamples'] = traces.shape[1]
        cluster_table['ss_batch_name'] = ss_dirname
        cluster_table['ss_batch_id'] = ss_i

        # merge in useful metadata from ephys mapping.csv
        mapping_cols = ['amplifier_id', 'mea1k_el', 'pad_id', 'metal', 'shank_name', 
                        'el_pair', 'mea1k_connectivity', 'connectivity_order',
                        'shank_side', 'curated_trace', 'depth', 'shank_id', 
                        'gross_brain_area', 'fine_brain_area']
        cluster_metad = mapping.loc[cluster_table.cluster_channel, mapping_cols].drop_duplicates()
        cluster_metad = cluster_metad.reset_index().rename({'index': 'cluster_channel',
                                                            'el_pair': 'fiber_id'}, axis=1)
        cluster_table = pd.merge(cluster_table, cluster_metad, on='cluster_channel')
        cluster_table = cluster_table[~np.isin(cluster_table.cluster_type, ('to_delete', ""))]
        L.logger.debug(f"Spike cluster table:\n{cluster_table}")
        return cluster_table

    def aggr_spike_cluster_metadata(cluster_aggr_over_ss_batches):
        # first process the spike metadata, merge them and create unique cluster ids, crucial
        # TODO add in session_wise spike count from spikes below? add average waveforms?
        # TODO add brain region to schema
        sp_clust_metadata = pd.concat(cluster_aggr_over_ss_batches, axis=0)
        
        unique_ids = sp_clust_metadata.cluster_id.unique() # unique because of + 10_000*ss_i
        # new ids start at 1 and go go up to the number of unique ids
        renamer = {old_id: new_id+1 for new_id, old_id in enumerate(unique_ids)}
        sp_clust_metadata['cluster_id'] = sp_clust_metadata['cluster_id'].replace(renamer)
        
        # assign colors per cluster    
        map_colors = np.vectorize(make_discr_cluster_id_cmap(sp_clust_metadata['cluster_id']).get)
        sp_clust_metadata['cluster_color'] = map_colors(sp_clust_metadata['cluster_id'])
        sp_clust_metadata['cluster_id_str'] = [f"Unit{c_id:04d}" for c_id in sp_clust_metadata['cluster_id']]
        return sp_clust_metadata

    def ssbatch_process_spikes(ss_res_fullfname, session_from_smple, session_to_smple):
        # open jrc output to read single spikes in a the interval matching the session
        with h5py.File(ss_res_fullfname, 'r') as f:
            ss_spike_times = f['spikeTimes'][0]
            L.logger.debug(f"Session samples within concatenated data conext go"
                        f" from {session_from_smple:,} to {session_to_smple:,}." 
                        f"All spike ({len(ss_spike_times):,}) go from sample {ss_spike_times[0]:,} to {ss_spike_times[-1]:,}.")
                        # f" This session's spikes are at: [{ss_spike_times[ss_from_spike_i]:,} ... {ss_spike_times[ss_to_spike_i]:,}]")
            
            
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
                nsites = 2
            else: 
                spike_sites_3rd = f['spikeSites3'][0, ss_from_spike_i:ss_to_spike_i]
                nsites = 3
        
        spike_table = pd.DataFrame({
            "sample_id": spike_times-session_from_smple,
            "ephys_timestamp": (spike_times-session_from_smple).astype(np.uint64)*50,
            "cluster_id_ssbatch": spike_clusters.astype(int),
            "channel": spike_sites-1, # matlab 1-based index
            "amplitude_uV": spike_amps /6.3, # error in JRC settings, reset here, TODO rm later
            "ss_batch_id": ss_i,
            # "channel_hpf_wf": pd.NA,
            # "channel_2nd_hpf_wf": pd.NA,
            # "channel_3rd_hpf_wf": pd.NA,
            "channel_2nd": spike_sites_2nd-1, # matlab 1-based index
            "channel_3rd": spike_sites_3rd-1, # matlab 1-based index
            "shank": np.round(spike_positions)//1000, # convert back from original shank * 1000
        })
        
        # remove hard deleted clusters, using backspace in JRC (-1)
        spike_table = spike_table.loc[spike_table.cluster_id_ssbatch!=-1].reset_index(drop=True)

        # # TODO this was neccessary before
        # print(spike_table)
        # if get_spikes_cluster_subset is not None:
        #     print(get_spikes_cluster_subset)
        #     incl_clust = get_spikes_cluster_subset[get_spikes_cluster_subset.ss_batch_id == ss_i].cluster_id_ssbatch
        #     spike_table = spike_table[spike_table.cluster_id_ssbatch.isin(incl_clust)]
        # print(spike_table)
        # exit()
        
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
        return spike_table, nsites
        
    def aggr_spikes(spike_aggr_over_ss_batches, sp_clust_metadata):    
        # after metadata is processed, we can process the single spikes
        spikes = pd.concat(spike_aggr_over_ss_batches, axis=0)

        spikes.sort_values(by='sample_id', inplace=True) # to be sure

        # use processed sp_clust_metadata to assign unique cluster ids to spikes
        # create a mapping between non_unique cluster ids from seperate spike sorting 
        # batches to unique ids in metadata
        unq_id_renamer = (sp_clust_metadata.loc[:, ['cluster_id_ssbatch', 'ss_batch_id']].values).tolist()
        unq_id_renamer = {str(k): c_id for k, c_id in zip(unq_id_renamer, sp_clust_metadata.cluster_id)}

        # use the mapping to assign unique cluster ids to single spikes
        keys = [str(k) for k in spikes.loc[: , ['cluster_id_ssbatch', 'ss_batch_id']].values.tolist()]
        spikes['cluster_id'] = [unq_id_renamer.get(k, 0) for k in keys]
        spikes['cluster_id_str'] = spikes['cluster_id'].apply(lambda x: f"Unit{x:04d}")

        # add cluster color for each spike
        cols = sp_clust_metadata.set_index('cluster_id').reindex(spikes['cluster_id']).cluster_color
        spikes['cluster_color'] = cols.fillna('#888888').values # needed bc of cl_id == 0 or not in metadata yet (sorted)
        
        # add brain region
        spikes = pd.merge(spikes, sp_clust_metadata[['cluster_id', 'gross_brain_area', 'fine_brain_area']],
                          on='cluster_id', how='left')
        spikes = spikes[np.isin(spikes.cluster_id, sp_clust_metadata.cluster_id)]
        return spikes.reset_index(drop=True)
    
    def extract_waveforms(spike_table, nsites):
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
                    all_filt_wfs[site_k].append(filt_wf[filt_window//2 -10:-filt_window//2 +25])
            # print(f"Filtered waveforms took {time.time()-t2:.3f} s")
            
            # instert spike waveform into spike_table
            all_filt_wfs_aggr = []
            for site_k, col_name in enumerate(chnls_str):
                all_filt_wfs_aggr.append(pd.Series(all_filt_wfs[site_k], 
                                                    name=col_name+'_hpf_wf',))
            return pd.concat(all_filt_wfs_aggr, axis=1)
        
        # # skip this if don't want to wait 20 minutes per sessions
        # TODO parallelize this once server reads are concurrent
        L.logger.debug(f"Extracting wavefroms & filtering...")
        nspikes_per_read = 10
        filt_window = 512
        chnls_str = ['channel', 'channel_2nd', 'channel_3rd'] if nsites == 3 else ['channel', 'channel_2nd']
        
        sos_filter = butter(4, [300, 5000], btype='band', fs=20_000)
        start_t = time.time()
        # for chunk_i in range(spike_table.shape[0] //nspikes_per_read):
        #     from_i, to_i = chunk_i*nspikes_per_read, (chunk_i+1)*nspikes_per_read
        for from_i in spike_table.index[::nspikes_per_read]:
            to_i = min(from_i + nspikes_per_read, spike_table.shape[0])
            waveforms = process_spike_chunk(spike_table.loc[from_i:to_i].copy(),
                                            sos_filter)
            # add the waveforms to the spike table
            spike_table.loc[from_i:to_i, waveforms.columns] = waveforms.values
            
            # if from_i > 1000:
            #     break
            
            sp_per_second = to_i /(time.time()-start_t)
            print(f"{to_i:07_d}/{spike_table.shape[0]:_} spikes, "
                    f"{(spike_table.shape[0]-to_i) /sp_per_second /60:.2f} "
                    f"minutes left at {sp_per_second:.2f} spikes/s", 
                    end='\r' if to_i < spike_table.shape[0] else 
                    f'\ntook {(time.time()-start_t)/60:.2f}min')
        return spike_table

    def calc_waveforms_infos(spikes, nsites):
        L.logger.debug(f"Calculating waveform infos for {spikes.cluster_id.nunique():,} clusters")
        def calc_wf_info(wf_df):
            channel_counts = {}
            # iterate over 1. 2. 3. bubble channel and count how often a pike occurs on there
            for chnl_str in chnls_str:
                wf_count = wf_df[chnl_str].value_counts().to_dict()
                # make keys strings
                wf_count = {f"{k}": v for k, v in wf_count.items()}
                channel_counts[f'{chnl_str}_wf_count'] = wf_count
            
            # make a flat list channl_ids and their waveforms
            all_chnls = wf_df[chnls_str].values.flatten()
            all_wfs = np.stack(wf_df[chnls_wfs_str].values.flatten())
            
            m, std = {}, {}
            # for each channel, no matter if 1. 2. or 3. in bubble, calculate the mean and std
            for chnl in np.unique(all_chnls):
                mask = all_chnls == chnl
                mean_wf = np.round(np.mean(all_wfs[mask], axis=0).astype(np.int16))
                std_wf = np.round(np.std(all_wfs[mask], axis=0).astype(np.int16))
                m[f"{chnl}"] = mean_wf
                std[f"{chnl}"] = std_wf

            avg_std_wf = {'averge_wf': m, 'std_wf': std}
            return channel_counts, avg_std_wf

        # set the bubble channel str column names
        chnls_str = ['channel', 'channel_2nd', 'channel_3rd'] if nsites == 3 else ['channel', 'channel_2nd']
        chnls_wfs_str = [f"{c}_hpf_wf" for c in chnls_str]
        
        cluster_infos = []
        for cluster_id, wf_df in spikes[['cluster_id', *chnls_str, *chnls_wfs_str]].groupby('cluster_id'):
            clst_chnl_counts, avg_std_wf = calc_wf_info(wf_df)
            cluster_infos.append(pd.Series({**clst_chnl_counts, **avg_std_wf, 
                                            **{'cluster_id': cluster_id,}}))
        return  pd.concat(cluster_infos, axis=1).T
    






    # ==========================================================================
    nas = device_paths()[0]
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    animal_name = session_name.split("_")[2]
    ss_info = pd.read_csv(os.path.join(nas, "devices", "animal_meta_info.csv"), 
                          index_col='animal_name')
    
    ss_dirnames = ss_info.loc[animal_name, 'ss_dirnames'].replace(" ", "").split(",")
    L.logger.debug(f"Found {len(ss_dirnames)} spike sorting runs/ batches.")
    traces, mapping = ml.session_modality_from_nas(session_fullfname, key='ephys_traces')
    if mapping is None:
        return None, None
    # print(f"Mapping:\n{mapping}")   

    ## spike sorting batches are compleltely seperate JRC runs/ directories, often split by region
    spike_aggr_over_ss_batches = []
    cluster_aggr_over_ss_batches = []
    for ss_i, ss_dirname in enumerate(ss_dirnames):
        ss_fulldir = os.path.join(nas, f"RUN_{animal_name}", "concatenated_ss", ss_dirname)
        L.logger.debug(f"Processing spikes from {ss_fulldir} for {session_name}")
        
        ss_session_lengths = pd.read_csv(os.path.join(ss_fulldir, "concat_session_lengths.csv"))
        ss_session_names = ss_session_lengths['name'].apply(lambda x: x[:x.find("min_")+3]).values
        L.logger.debug(f"Containes spike sorted sessions:\n{ss_session_names}")
        
        # unpack the concateated JRC output, going back to single session sample indices 
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
        
        # get the cluster metadata
        cluster_table = ssbatch_cluster_metadata(ss_res_fullfname)
        cluster_aggr_over_ss_batches.append(cluster_table)
            
        # get the spikes
        spike_table, nsites = ssbatch_process_spikes(ss_res_fullfname, session_from_smple, 
                                                     session_to_smple)
        spike_aggr_over_ss_batches.append(spike_table)
        L.spacer("debug")
    
    # agggregate over batches and handle unique cluster ids
    sp_clust_metadata = aggr_spike_cluster_metadata(cluster_aggr_over_ss_batches)
    L.logger.debug(f"Aggregation over spike sorting groups:\n{sp_clust_metadata}")
    
    # aggregate spikes over batches, using the metadata to assign unique cluster ids
    spikes = aggr_spikes(spike_aggr_over_ss_batches, sp_clust_metadata)
    L.logger.debug(f"Aggregated spikes:\n{spikes}")
    # extract waveforms from raw traces.dat and insert into spikes dataframe
    spikes = extract_waveforms(spikes, nsites)
    
    # cluster metadata with session spike counts, and avg waveforms
    nspikes = spikes.cluster_id.value_counts().reindex(sp_clust_metadata.cluster_id, fill_value=0)
    sp_clust_metadata['session_spike_count'] = nspikes.values
    # use exteracted waveforms to calculate waveform infos, like average and std, channel wise
    waveform_infos = calc_waveforms_infos(spikes, nsites)
    sp_clust_metadata = pd.merge(sp_clust_metadata, waveform_infos, on='cluster_id', 
                                 how='left')
    return sp_clust_metadata, spikes








































def get_FiringRate40msHz(spikes, all_cluster_names):
    # create a timer interval calmun from sample_id
    bin_size_us_us = 40_000 # in us
    breaks = np.arange(0, spikes['ephys_timestamp'].max()+bin_size_us_us, 
                       bin_size_us_us, dtype='uint64')
    spikes['bin_40ms'] = pd.cut(
        spikes['ephys_timestamp'],
        bins=pd.IntervalIndex.from_breaks(breaks)
    )
    
    # then group by clusterID and use the bin_40ms to count spikes
    cnt = lambda x: x['bin_40ms'].value_counts() * 25 # 25 Hz = 1/40ms
    fr_40ms_hz = spikes.groupby('cluster_id_str').apply(cnt, include_groups=False)
    fr_40ms_hz = fr_40ms_hz.unstack().fillna(0).T
    # reindex to all cluster_ids across sessions
    fr_40ms_hz = fr_40ms_hz.reindex(all_cluster_names.iloc[:,0].values, fill_value=0, axis=1)
    
    # append the bin edges    
    fr_40ms_hz['from_ephys_timestamp'] = breaks[:-1]
    fr_40ms_hz['to_ephys_timestamp'] = breaks[1:]
    
    return fr_40ms_hz.reset_index(drop=True)

def get_FiringRate40msZ(fr_hz):
    def zscore(x):
        if not x.name.startswith('Unit'):
            return x
        if (x == 0).all():
            return np.nan * np.ones_like(x)
        z = ((x - x.mean()) / x.std())
        return z
    return fr_hz.apply(zscore, axis=0)

# compute_primitives.py import as comp_prim
def run_pca(Z, skip_projection=False):
    corr_matrix = np.corrcoef(Z)
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    eigenval_sum = np.sum(eigenvals)
    expl_var = eigenvals / eigenval_sum
    
    # # calculate projection of the data onto the eigenvectors
    if not skip_projection:
        Z_proj =  eigenvecs @ Z
        Z_proj = pd.DataFrame(Z_proj.T, # index=Z.index,
                              columns=[f"PC{i+1}" for i in range(Z_proj.shape[0])])
    else:
        Z_proj = None
        
    PCs = pd.DataFrame(eigenvecs, columns=[f"PC{i+1}" for i in range(eigenvecs.shape[1])])
    PCs = pd.concat([PCs, pd.Series(eigenvals, name='eigenvalues')], axis=1)
    PCs = pd.concat([PCs, pd.Series(expl_var, name='explained_variance')], axis=1)
    return PCs, Z_proj
    
    
    
    
    
    
    
    
    
def get_sessionPCA(fr_z):
    fr_z = fr_z.groupby(level='session_id').apply(lambda x: x.fillna(x.min().min()))
    fr_z = fr_z.set_index(["from_ephys_timestamp","to_ephys_timestamp"], append=True)
    fr_z.reset_index(level=(0,1,2,4), inplace=True, drop=True)
    
    # lst = fr_z.iloc[-1] 
    # fr_z = fr_z.iloc[:25*16*5]
    # fr_z.iloc[-1] = 0.0001
    
    print(f"Firing rate shape: {fr_z.shape}")
    Z = fr_z.values.T  # transpose to have neurons in rows, time bins in columns
    
    # PCs, Z_proj = comp_prim.run_pca(Z)
    PCs, Z_proj = run_pca(Z)
    
    PCs['cluster_id_str'] = fr_z.columns.values
    Z_proj['from_ephys_timestamp'] = fr_z.index.get_level_values('from_ephys_timestamp')
    Z_proj['to_ephys_timestamp'] = fr_z.index.get_level_values('to_ephys_timestamp')
    return PCs, Z_proj

def get_SessionPCs40msCAs(PCs):
    angle_aggr = []
    # Use distinct variable names for the two session comparisons
    session_ids = PCs.index.unique(level='session_id')
    Logger().logger.debug(f"Calculating canonical angles between "
                          f"{len(session_ids)} session PC spaces...")
    for from_s_id in session_ids:
        # Get the PCs for the "from" session and keep only numeric PC columns
        from_session_pc = PCs.xs(from_s_id, level='session_id')
        # we align the subspace of the n components that explain 80% of the variance
        from_session_nPCs = (from_session_pc.explained_variance.cumsum()<.8).sum()
        from_s_subspace = from_session_pc.filter(regex='^PC').values[:, :from_session_nPCs]
        
        for to_s_id in session_ids:
            # Get the PCs for the "to" session
            to_session_pc = PCs.xs(to_s_id, level='session_id')
            to_session_nPCs = (to_session_pc.explained_variance.cumsum()<.8).sum()
            to_s_subspace = to_session_pc.filter(regex='^PC').values[:, :to_session_nPCs]
            # print(f"Comparing S{from_s_id}({from_session_nPCs}): to S{to_s_id}({to_session_nPCs})")
            
            # Compute SVD between matching subspaces
            M = from_s_subspace.T @ to_s_subspace
            from_s_c, S, to_s_comp_h_c = np.linalg.svd(M)
            canonc_angles = np.arccos(np.clip(S, -1, 1))
            canonc_angles = np.round(np.rad2deg(canonc_angles), 2)
            angle_aggr.append(pd.Series(canonc_angles, name=(from_s_id, to_s_id),
                                        index=[f"CA{ca}" for ca in range(canonc_angles.shape[0])],
                                        ))
    angle_aggr = pd.concat(angle_aggr, axis=1).T
    angle_aggr.index.names = ['from_session_id', 'to_session_id']
    return angle_aggr.reset_index()











def get_Ensemble40msProjEventAligned(ensemble_proj, behavior):
    # convert session_id to a column in behavior, make ephys timestamp the index
    behavior['session_id'] = behavior.index.get_level_values('session_id')
    behavior.reset_index(inplace=True, drop=True)
    behavior.set_index('frame_ephys_timestamp', inplace=True)
    
    # index by ephys timestamp intervals
    ensemble_proj.index = pd.IntervalIndex.from_arrays(ensemble_proj.pop("from_ephys_timestamp"),
                                                       ensemble_proj.pop("to_ephys_timestamp"))

    bin_size_us = ensemble_proj.index[0].right - ensemble_proj.index[0].left
    
    all_sess_aggr = []
    for s_id in ensemble_proj.session_id.unique():
        sess_behavior = behavior[behavior.session_id == s_id].drop(columns='session_id')
        sess_ensemble_proj = ensemble_proj[ensemble_proj.session_id == s_id].drop(columns='session_id')
        
        print(f"\nProcessing session {s_id} "
                f"from bin {sess_ensemble_proj.shape} bins.", flush=True)


        # assign each row/timestamp to its 40ms interval bin
        sess_behavior['ephys_bin'] = sess_ensemble_proj.index.get_indexer(sess_behavior.index)

        # trial ids very sparsly have errors, filter out invalid values
        categoc_agg = sess_behavior[['ephys_bin', 'trial_id']].groupby('ephys_bin').agg(lambda x: x.mode())
        valid_trial_mask = [True if isinstance(v, np.int16) and v >= 1 else False for v in categoc_agg['trial_id']]
        categoc_agg[~np.array(valid_trial_mask)] = np.nan
        
        # then group by this index and take the mean
        sess_behavior = sess_behavior.groupby('ephys_bin').mean().round(3)
        # overwrite the mean with the mode agg for categorical variables
        sess_behavior['trial_id'] = categoc_agg['trial_id']
        # drop -1 bin
        sess_behavior = sess_behavior.loc[sess_behavior.index >= 0]
        # slice to bins for which there is behavior (eg missing before unity launches)
        sess_ensemble_proj = sess_ensemble_proj.iloc[sess_behavior.index]
        sess_behavior.index = sess_ensemble_proj.index.mid.values.astype(np.int64)
        sess_ensemble_proj.index = sess_behavior.index

        # fig, axs = plt.subplots(nrows=2, figsize=(5, 11))
        # axs[1].scatter(sess_behavior['frame_raw'], sess_ensemble_proj['Assembly003'],  alpha=.2, )
        # plt.savefig("quickviz.png")

        # print(sess_behavior)
        # print(sess_ensemble_proj)
        # zones = {
        #     'beforeCueZone': (-168, -100),
        #     'cueZone': (-80, 25),
        #     'afterCueZone': (25, 50),
        #     'reward1Zone': (50, 110),
        #     'betweenRewardsZone': (110, 170),
        #     'reward2Zone': (170, 230),
        #     'postRewardZone': (230, 260),
        #     'wholeTrack': (-168, 260),
        # }
        zones = {
            'enter_cueZone': -80,
            'enter_afterCueZone': 25,
            'enter_reward1Zone': 50,
            'enter_reward2Zone': 170,
        }
        zones_t_interval_s = {
            'enter_cueZone': 1,
            'enter_afterCueZone': 1,
            'enter_reward1Zone': 1,
            'enter_reward2Zone': 1,
        }


        
        
        # kinematics
        # TODO find t points of specific acceleration events, and deceleration events
        # acc = sess_behavior['frame_acceleration']
        # acc.loc[:] = np.clip(acc, -15, 15)

        # TODO find t points of specific acceleration events, and deceleration events also here
        # # shift down the ephys index by one second, predict future acceleration
        # forward_acc_in1sec = acc.copy()
        # nbins_1sec = int(1 * 1_000_000 / bin_size_us)
        # idx = forward_acc_in1sec.index.values[:-nbins_1sec]
        # forward_acc_in1sec = forward_acc_in1sec.iloc[nbins_1sec:]
        # forward_acc_in1sec.index = idx
        
        def select_t0_events(trial_beh):
            # print(trial_beh)
            base_info = trial_beh[['cue', 'trial_outcome', 'choice_R1', 'choice_R2']].iloc[0].to_dict()

            zone_ts = []
            for zone_name, zone_x in zones.items():
                t = np.abs(trial_beh['frame_position']-zone_x).sort_values()
                if t.iloc[0]<3: # a valid location should be within 3 cm of the zone
                    t = t.index[0]
                else:
                    print(f"Warning: No t0 event found for {zone_name}, was {t.iloc[0]} cm away")
                    continue    
                interv_s = zones_t_interval_s[zone_name]
                interv_us = pd.Interval(left=t-interv_s*1_000_000, 
                                        right=t+interv_s*1_000_000,)

                zone_ts.append({**base_info, 't0_event_name': zone_name, 
                                't0': t, 't_interval': interv_us})
            zone_ts = pd.DataFrame(zone_ts)
            return zone_ts
        
        # make a table where every row is a t0 event, keep info like cue, outcome, choice, trial_id
        t0_events = sess_behavior.groupby('trial_id').apply(select_t0_events, 
                                                    include_groups=False).reset_index().drop(columns='level_1')








        #TODO should be somewhere else
        def make_non_overlapping_intervals(intervals):
            # make sure the intervals are non-overlapping, 
            # if they are overlapping, add them the iloc to a new list
            # make new lists if there is no list that wouldn't have overlapping intervals
            
            def merge_interval_indices(index1, index2):
                # Extract the left and right endpoints from both indices
                left_endpoints = list(index1.left) + list(index2.left)
                right_endpoints = list(index1.right) + list(index2.right)
                
                # Create a new IntervalIndex from the combined endpoints
                return pd.IntervalIndex.from_arrays(
                    left_endpoints,
                    right_endpoints,
                    closed=index1.closed  # Use the closed parameter from the first index
                )
            
            def add_interval_to_batch(interv):
                batch_id = 0
                while True:
                    # create a new batch if the current one doesn't exist yet
                    if len(non_overlapping_interv_batches) <= batch_id:
                        # init with a single interval
                        non_overlapping_interv_batches.append(interv)
                        # print(f"Batch {batch_id} is empty, adding interval {interv}, {type(interv)}")
                        return batch_id
                    
                    else:
                        hpyth_index = merge_interval_indices(
                            non_overlapping_interv_batches[batch_id], interv
                        )
                    # check if the new interval overlaps with the current batch
                    if not hpyth_index.is_overlapping:
                        # if not, the merged interval becomes the new batch
                        non_overlapping_interv_batches[batch_id] = hpyth_index
                        # print(f"Batch {batch_id} is not overlapping, adding interval {interv}")
                        return batch_id
                    
                    else:
                        # if it does, create a new batch and try again
                        batch_id += 1
                        # print(f"Batch {batch_id} is overlapping, creating a new one")
                        # print(non_overlapping_interv_batches)
                        # print(interv)
                        # print()

            non_overlapping_interv_batches = []
            batch_assinments = {}
            for i in range(len(intervals)):
                cur_interv = intervals[i]
                batch_id = add_interval_to_batch(pd.IntervalIndex([cur_interv]))
                batch_assinments[i] = batch_id

            lengths = [len(batch) for batch in non_overlapping_interv_batches]
            print(f"Made {len(non_overlapping_interv_batches)} non-overlapping"
                    f" batches of intervals of lengths: {lengths}")
            return pd.Series(batch_assinments, name='interval_batch_id',)
                
                
                
                
                
                
                
                
                
        intervals = pd.IntervalIndex(t0_events.t_interval)
        # intervals cannot overlap for followup analysis, so we need to make 
        # batches of non-overlapping intervals
        batch_assinments = make_non_overlapping_intervals(intervals)
        t0_events = pd.merge(t0_events, batch_assinments, left_index=True, right_index=True)
        
        aggr = []
        for interv_batch_id in t0_events.interval_batch_id.unique():
            # get the t0 events for this batch, reset the index for merging later
            t0_events_batch = t0_events[t0_events.interval_batch_id == interv_batch_id].reset_index(drop=True)
            t0_events_batch.index.name = 'event_assignm'
            print(f"Processing t0 events batch {interv_batch_id} with {len(t0_events_batch)} events")
            
            # use the intervals to assign each ephys t point to a t0 event
            intervals = pd.IntervalIndex(t0_events_batch.t_interval)
            event_assignm = intervals.get_indexer(sess_ensemble_proj.index)
            
            t0_ensemble_proj_batch = sess_ensemble_proj.copy()
            t0_ensemble_proj_batch['event_assignm'] = event_assignm
            t0_ensemble_proj_batch[t0_ensemble_proj_batch['event_assignm'] != -1]
            # put the ephys t and event assignm into columns, reset the index
            t0_ensemble_proj_batch = t0_ensemble_proj_batch.reset_index().rename(columns={'index': 'ephys_bin_t'})

            # groupby the event assigment and stack the ephys t points
            # length depdends on the interval; Assemblies are columns, + one t column
            t0_ensemble_proj_batch = t0_ensemble_proj_batch.groupby('event_assignm').apply(lambda x:
                                            (x.reset_index(drop=True).stack()),
                                            include_groups=False).unstack(level=2)
            # print(t0_ensemble_proj_batch)
            t0_ensemble_proj_batch.index.names = ['event_assignm', 'interval_t']

            # merge behavior and ensemble projection, using event assignm as index
            t0_events_batch = pd.merge(t0_events_batch, t0_ensemble_proj_batch, 
                                        left_index=True, right_index=True, how='left')
            # keep only the t_interval (eg 50 points for 2 seconds, 40ms bins)
            t0_events_batch = t0_events_batch.reset_index().drop(columns='event_assignm')
            aggr.append(t0_events_batch)
            
        # concat all t0 events batches
        sess_t0_events = pd.concat(aggr, axis=0).sort_values(['t0','interval_t']).reset_index(drop=True)
        sess_t0_events['session_id'] = s_id
        all_sess_aggr.append(sess_t0_events)
        # print(all_sess_aggr)
        
    all_sess_aggr = pd.concat(all_sess_aggr, axis=0).reset_index(drop=True)
    print(all_sess_aggr)
    return all_sess_aggr
        


def get_TrackwiseEnsembleProj(ens_proj: pd.DataFrame,
                             track_behavior_data: pd.DataFrame) -> pd.DataFrame:
    """
    Trackwise (binned) ensemble projection, aligned to BehaviorTrackwise bins.
    Animal-level safe: processes each session separately to avoid cross-session mixing.
    """

    # --- split by session to avoid mixing bins across sessions ---
    if "session_id" not in ens_proj.columns:
        raise ValueError("ens_proj must contain a 'session_id' column for animal-level processing.")

    sess_ids_beh = track_behavior_data.index.get_level_values("session_id").unique()
    sess_ids_ens = pd.Index(ens_proj["session_id"].unique())
    sess_ids = [s for s in sess_ids_beh if s in set(sess_ids_ens)]

    all_sessions = []

    for s_id in sess_ids:
        beh_s = track_behavior_data.xs(s_id, level="session_id").copy()
        ens_s = ens_proj.loc[ens_proj["session_id"] == s_id].copy()

        # recover interval index for this session only
        ens_s.index = pd.IntervalIndex.from_arrays(
            ens_s.pop("from_ephys_timestamp"),
            ens_s.pop("to_ephys_timestamp"),
        )
        ens_s = ens_s.drop(columns=["session_id"], errors="ignore")

        def time_bin_avg(posbin_data: pd.DataFrame) -> pd.DataFrame:
            # remove rows that cannot define intervals
            posbin_data = posbin_data.dropna(
                subset=["posbin_from_ephys_timestamp", "posbin_to_ephys_timestamp"]
            )
            if posbin_data.empty:
                return pd.DataFrame()

            # convert to real NumPy ints (avoids nullable Int64 masked arrays)
            from_posbin_t = posbin_data["posbin_from_ephys_timestamp"].to_numpy(dtype="int64")
            to_posbin_t   = posbin_data["posbin_to_ephys_timestamp"].to_numpy(dtype="int64")

            interval = pd.IntervalIndex.from_arrays(
                from_posbin_t - 40_000,
                to_posbin_t + 40_000,
                closed="both",
            )

            # assign each 40ms ensemble bin midpoint to a trial in this position bin
            assigned_bin = pd.cut(
                ens_s.index.mid,
                bins=interval,
                labels=posbin_data.trial_id[:-1],
            )
            trials_exist_mask = (assigned_bin.value_counts() != 0).values

            # if nothing overlaps, return empty
            if assigned_bin.notna().sum() == 0:
                return pd.DataFrame()

            # keep only overlapping bins
            trial_proj = ens_s.loc[assigned_bin.notna()].copy().astype(np.float32)
            trial_proj["posbin_t_edges"] = assigned_bin[assigned_bin.notna()]

            posbin_trial_wise = trial_proj.groupby("posbin_t_edges", observed=True).mean()

            # align metadata to the set of trials that actually exist
            # vc = assigned_bin.value_counts()
            # trials_exist = vc.index.to_numpy()
            # trials_exist_mask = posbin_data["trial_id"].isin(trials_exist).to_numpy()

            posbin_trial_wise["cue"] = posbin_data.loc[trials_exist_mask, "cue"].to_numpy()
            posbin_trial_wise["trial_outcome"] = posbin_data.loc[trials_exist_mask, "trial_outcome"].to_numpy()
            posbin_trial_wise["choice_R1"] = posbin_data.loc[trials_exist_mask, "choice_R1"].to_numpy()
            posbin_trial_wise["choice_R2"] = posbin_data.loc[trials_exist_mask, "choice_R2"].to_numpy()
            posbin_trial_wise["bin_length"] = interval.length[trials_exist_mask] / 1e6

            posbin_trial_wise.index = posbin_data.loc[trials_exist_mask, "trial_id"].to_numpy()
            return posbin_trial_wise

        # group within session only
        sess_out = beh_s.groupby("from_position_bin").apply(time_bin_avg)
        if isinstance(sess_out, pd.DataFrame) and len(sess_out) > 0:
            sess_out.index = sess_out.index.rename(["from_position_bin", "trial_id"])
            sess_out = sess_out.reset_index(drop=False)
            sess_out["session_id"] = s_id
            all_sessions.append(sess_out)

    if len(all_sessions) == 0:
        return pd.DataFrame()

    return pd.concat(all_sessions, axis=0, ignore_index=True)


def get_FiringRateTrackwiseEnsemble(fr_trackwise, ens_weights):

    def norm_unit(u):
        s = str(u)
        if s.startswith("Unit"):
            m = re.fullmatch(r"Unit0*(\d+)", s)
            return f"Unit{int(m.group(1)):04d}" if m else s
        if s.isdigit():
            s = int(s)+1
            return f"Unit{int(s):04d}"
        return s

    ignore_cols = ['from_position_bin','trial_id','cue','trial_outcome','choice_R1','choice_R2','bin_length']
    ignore_cols = [c for c in fr_trackwise.columns if c in ignore_cols]

    units = [c for c in fr_trackwise.columns if isinstance(c, str) and c.startswith("Unit")]
    X = fr_trackwise[units]
    # z-scoring firing rates
    X = (X - X.mean()) / X.std()
    W = ens_weights.copy()
    W.index = [norm_unit(i) for i in W.index]
    W = W.loc[units]
    X = X.apply(pd.to_numeric, errors="coerce")
    W = W.apply(pd.to_numeric, errors="coerce")

    Xv = np.nan_to_num(X.to_numpy(), nan=0.0)
    Wv = np.nan_to_num(W.to_numpy(), nan=0.0)
    Y = Xv @ Wv
    diag = (Xv**2) @ (Wv**2)
    quad_coact = Y**2 - diag
    calc = pd.DataFrame(quad_coact, index=fr_trackwise.index, columns=W.columns)

    ens_fr = pd.concat([calc, fr_trackwise[ignore_cols]], axis=1)

    #renaming for plotting
    # ens_fr = ens_fr.rename(columns=lambda c: re.sub(r"^Assembly(\d+)$", r"Unit0\1", str(c)))

    return ens_fr
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    #         # before
    #         bef_ephys = smoothed.to_frame().copy()
    #         bef_ephys['event_i'] = one_y_tpoints.get_indexer(bef_ephys.index)
    #         # get the ephys before a single event and rest to count index (order before event)
    #         bef_ephys = bef_ephys[bef_ephys['event_i'] != -1].groupby('event_i').apply(lambda x: x.reset_index(drop=True))
    #         # then use this order to groupby the single timepoints before the event
    #         bef_ephys = bef_ephys.drop(columns='event_i').unstack()
    #         # add the input info, like cue outcome choice
    #         bef_ephys['input'] = y.values
    #         print()
    #         print()
    #         print(bef_ephys)
    #         print(y)
    #         bef_ephys.set_index('input', inplace=True, append=True)
            
    #         # the same for after the event
    #         aft_ephys = smoothed.to_frame().copy()
    #         aft_ephys['event_i'] = other_y_tpoints.get_indexer(aft_ephys.index)
    #         # get the ephys after a single event and rest to count index (order after event)
    #         aft_ephys = aft_ephys[aft_ephys['event_i'] != -1].groupby('event_i').apply(lambda x: x.reset_index(drop=True))
    #         # then use this order to groupby the single timepoints after the event
    #         aft_ephys = aft_ephys.drop(columns='event_i').unstack()
    #         # add the input info, like cue outcome choice
    #         print(aft_ephys)
    #         print(y)
    #         aft_ephys['input'] = y.values
    #         aft_ephys.set_index('input', inplace=True, append=True)
            
            
            
    #         exit()
            
    #         # locations per trial, find the in a trial where the x position is closest to the zone entry or exit
    #         trial_cue_entry_t = sess_behavior.groupby('trial_id').apply(lambda x: 
    #                                 np.abs(x['frame_position']-zones['cueZone'][0]).sort_values().index[0])
    #         trial_cue_entry_t = trial_cue_entry_t.reset_index().set_index(0).squeeze() # flip index and values, t is index again
            
    #         trial_postcue_entry_t = sess_behavior.groupby('trial_id').apply(lambda x: 
    #                                 np.abs(x['frame_position']-zones['cueZone'][1]).sort_values().index[0])
    #         trial_postcue_entry_t = trial_postcue_entry_t.reset_index().set_index(0).squeeze() # flip index and values, t is index again
            
    #         trial_R1entry_t = sess_behavior.groupby('trial_id').apply(lambda x: 
    #                                 np.abs(x['frame_position']-zones['reward1Zone'][0]).sort_values().index[0])
    #         trial_R1entry_t = trial_R1entry_t.reset_index().set_index(0).squeeze() # flip index and values, t is index again

    #         trial_R2entry_t = sess_behavior.groupby('trial_id').apply(lambda x: 
    #                                 np.abs(x['frame_position']-zones['reward2Zone'][0]).sort_values().index[0])
    #         trial_R2entry_t = trial_R2entry_t.reset_index().set_index(0).squeeze() # flip index and values, t is index again

    #         print('---------')
    #         # print(sess_behavior.index)
    #         print()
            
    #         trial_postcue_entry_t_outcome = sess_behavior.loc[trial_postcue_entry_t.index, 'trial_outcome']>1
    #         trial_postcue_entry_t_losses = trial_postcue_entry_t_outcome[trial_postcue_entry_t_outcome == False]
    #         trial_postcue_entry_t_wins = trial_postcue_entry_t_outcome[trial_postcue_entry_t_outcome == True]
            
    #         trial_R1entry_t_outcomes = sess_behavior.loc[trial_R1entry_t.index, 'trial_outcome']>1
    #         trial_R1entry_t_losses = trial_R1entry_t_outcomes[trial_R1entry_t_outcomes == False]
    #         trial_R1entry_t_wins = trial_R1entry_t_outcomes[trial_R1entry_t_outcomes == True]
            
    #         trial_R2entry_t_outcomes = sess_behavior.loc[trial_R2entry_t.index, 'trial_outcome']>1
    #         trial_R2entry_t_losses = trial_R2entry_t_outcomes[trial_R2entry_t_outcomes == False]
    #         trial_R2entry_t_wins = trial_R2entry_t_outcomes[trial_R2entry_t_outcomes == True]
            
    #         # exit()
    #         # print(trial_cue_entry_t.to_string())
            
    #         # print(sess_behavior.columns)
    #         # print(sess_behavior['frame_raw'])
    #         # exit()
            
            
    #         Y = pd.DataFrame({
                
    #         # kinematic variables
    #          'forward_vel':  sess_behavior['frame_raw'],
    #          'sideway_vel':  sess_behavior['frame_yaw'],
    #          'rotation_vel': sess_behavior['frame_pitch'],
             
    #          'forward_acc': acc,
    #          'forward_acc_abs': np.abs(acc),
    #          'forward_acc_positive_only': forward_acc_positive_only,
    #          'forward_acc_negative_only': forward_acc_negative_only,
            
    #          'forward_acc_in1sec': forward_acc_in1sec,
    #          'forward_acc_in1sec_abs': np.abs(forward_acc_in1sec),
    #          'forward_acc_in1sec_positive_only': forward_acc_positive_only_in1sec,
    #          'forward_acc_in1sec_negative_only': forward_acc_negative_only_in1sec,                
            
    #          # event based
    #          'lick': sess_behavior['lick_count']>0, # slice ephys later around event
    #          'reward_sound': sess_behavior['reward-sound_count']>0, # slice ephys later around event
    #          'reward_valve': sess_behavior['reward-valve-open_count']>0, # slice ephys later around event
             
    #          'cue_entry': trial_cue_entry_t,  # slice ephys later around this t
    #          'cue1_vs_cue2': sess_behavior.loc[trial_cue_entry_t.index, 'cue'], # slice ephys later around this t
             
    #          'cue1_vs_cue2_post': sess_behavior.loc[trial_postcue_entry_t.index, 'cue'], # slice ephys later around this t
    #          'cue1_vs_cue2_post_wins': sess_behavior.loc[trial_postcue_entry_t_wins.index, 'cue'], # slice ephys later around this t
    #          'cue1_vs_cue2_post_losses': sess_behavior.loc[trial_postcue_entry_t_losses.index, 'cue'], # slice ephys later around this t

    #         #  'post_cue_outcome': sess_behavior.loc[trial_postcue_entry_t.index, 'trial_outcome'], # slice ephys later around this t
    #         #  'post_cue1_outcome': sess_behavior.loc[trial_postcue_entry_t.index, 'cue'], # slice ephys later around this t
    #         #  'post_cue2_outcome': sess_behavior.loc[trial_postcue_entry_t.index, 'cue'], # slice ephys later around this t

    #          'R1choice_entry': sess_behavior.loc[trial_R1entry_t.index, 'choice_R1'], # slice ephys later around this t
    #          'R1choice_entry_wins': sess_behavior.loc[trial_R1entry_t_wins.index, 'choice_R1'], # slice ephys later around this t
    #          'R1choice_entry_losses': sess_behavior.loc[trial_R1entry_t_losses.index, 'choice_R1'], # slice ephys later around this t

    #          'R2choice_entry': sess_behavior.loc[trial_R2entry_t.index, 'choice_R2'], # slice ephys later around this t
    #          'R2choice_entry_wins': sess_behavior.loc[trial_R2entry_t_wins.index, 'choice_R2'], # slice ephys later around this t
    #          'R2choice_entry_losses': sess_behavior.loc[trial_R2entry_t_losses.index, 'choice_R2'], # slice ephys later around this t
    #          })

   
   
   
   
   
   
    #         print("-=------------=------")
   
    #         # for ensemble in [col for col in sess_ensemble_proj.columns if col.startswith("Assembly")]:
    #         for ensemble in sess_ensemble_proj.columns:
    #             if True:
    #             # if ensemble == 'Assembly001':
    #                 # Create a grid with 11 rows and 2 columns, but make right column 1/5 width of left

    #                 fig = plt.figure(figsize=(16, 16))
    #                 fig.suptitle(f'Session {s_id} - {ensemble}', fontsize=16, y=0.95)
    #                 gs = GridSpec(15, 2, width_ratios=[6, 1], figure=fig)
    #                 axs = np.empty((15, 2), dtype=object)
    #                 # Create first column with shared x/y axes
    #                 axs[0, 0] = fig.add_subplot(gs[0, 0])
    #                 for row in range(1, 15):
    #                     axs[row, 0] = fig.add_subplot(gs[row, 0], sharex=axs[0, 0], )
    #                 # Create second column with shared x/y axes
    #                 axs[0, 1] = fig.add_subplot(gs[0, 1])
    #                 for row in range(1, 15):
    #                     axs[row, 1] = fig.add_subplot(gs[row, 1], )#sharex=axs[0, 1], )
    #                 plt.subplots_adjust(hspace=0.001, wspace=0.001)
    #                 for ax_row in axs:
    #                     for ax in ax_row:
    #                         ax.set_xlabel('')
    #                         # ax.set_ylabel('')
    #                         ax.tick_params(axis='x', labelbottom=False)
    #                         # ax.tick_params(axis='y', labelleft=False)

    #             ephys = sess_ensemble_proj[ensemble].to_frame(name=ensemble)
    #             smoothed = ephys.iloc[:,0].rolling(window=6, center=True, min_periods=6).mean()
    #             axs[0, 0].plot(ephys.iloc[:,0].rolling(window=25*5, center=True, min_periods=25*5).mean().index, 
    #                            ephys.iloc[:,0].rolling(window=25*5, center=True, min_periods=25*5).mean().values, '-', markersize=1, alpha=1, color='black')
    #             axs[0, 0].set_ylim((-.4, 1.6))
                
    #             row = 1
    #             color_cycle = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
    #             for i, encodes_col in enumerate(Y.columns):
    #                 if (encodes_col.startswith('forward_') or encodes_col.startswith('rotation_') or encodes_col.startswith('sideway_')):
    #                     continue
                        
    #                 print(f'\n\n----{encodes_col}----')
    #                 y = Y[encodes_col]
                    
                        
                        
                        
                        
                        
                        
    #                 # events
    #                 if encodes_col in ['lick', 'reward_sound', 'reward_valve']:
    #                     y = y[y.values]
    #                 else:
    #                     y = y[y.notna()].astype(int)
    #                     print(y)
    #                     y = y[y.index<ephys.index[-1]-1_000_000]
    #                     print(y)
    #                 if y.empty:
    #                     continue
                    
    #                 # color = next(color_cycle)

    #                 # if encodes_col in ['reward_sound', ]:
    #                 #     axs[0, 0].scatter(y.index, np.ones_like(y)*smoothed.mean(), marker='|', s=1000, alpha=0.4, color=color)
    #                 axs[row, 0].set_title(f'{encodes_col}', fontsize=10, loc='left')
                    
    #                 # lick count
    #                 ymin, ymax = 0, 0
    #                 if encodes_col == 'lick':
    #                     axs[row, 0].scatter(y.index, np.zeros_like(y), marker='|', s=1000, alpha=0.6, color='blue')
    #                     y = y.groupby(y.index).count()
                        
    #                     # lick_tpoints = smoothed.copy()
    #                     other_y_ephys = smoothed.reindex(y.index)
    #                     one_y_ephys = smoothed.reindex(smoothed.index.difference(y.index))

    #                     axs[row, 1].scatter(-0.2*np.ones_like(other_y_ephys)+np.random.normal(scale=0.01, size=other_y_ephys.shape), 
    #                                         other_y_ephys.values, marker='.', s=1, alpha=.5, color='blue')
    #                     axs[row, 1].axhline(other_y_ephys.mean(), color='blue', linestyle='--', linewidth=1, alpha=0.8)
    #                     axs[row, 1].scatter(0.2*np.ones_like(one_y_ephys)+np.random.normal(scale=0.01, size=one_y_ephys.shape), 
    #                                         one_y_ephys.values, marker='.', s=1, alpha=.5, color='gray')
    #                     axs[row, 1].axhline(one_y_ephys.mean(), color='gray', linestyle='--', linewidth=3, alpha=0.2)
    #                     axs[row, 1].set_title(f'Lick count: {y.sum()}', fontsize=10, loc='left')
    #                     axs[row, 1].set_xlim(-0.5, 0.5)

    #                 # before after variables
    #                 if encodes_col in ['reward_sound', 'reward_valve', 
    #                                    'cue_entry', 'cue1_vs_cue2', 'cue1_vs_cue2_post',
    #                                    'cue1_vs_cue2_post_wins', 'cue1_vs_cue2_post_losses',
                                    
    #                                 'R1choice_entry', 'R1choice_entry_wins', 'R1choice_entry_losses',
    #                                 'R2choice_entry', 'R2choice_entry_wins', 'R2choice_entry_losses',
                                    
    #                                 ]:
                        
    #                     if encodes_col in ['reward_sound', 'reward_valve']:
    #                         interv_s = .5 # seconds, half a second before and after the event
    #                     else:
    #                         interv_s = 1
                        
    #                     one_y_tpoints = pd.IntervalIndex.from_arrays(y.index.values-interv_s*1_000_000, # half a second before the event, us
    #                                                                  y.index.values)
    #                     other_y_tpoints = pd.IntervalIndex.from_arrays(y.index.values,
    #                                                                    y.index.values+interv_s*1_000_000,) # half a second after the event, us
                        
    #                     # before
    #                     bef_ephys = smoothed.to_frame().copy()
    #                     bef_ephys['event_i'] = one_y_tpoints.get_indexer(bef_ephys.index)
    #                     # get the ephys before a single event and rest to count index (order before event)
    #                     bef_ephys = bef_ephys[bef_ephys['event_i'] != -1].groupby('event_i').apply(lambda x: x.reset_index(drop=True))
    #                     # then use this order to groupby the single timepoints before the event
    #                     bef_ephys = bef_ephys.drop(columns='event_i').unstack()
    #                     # add the input info, like cue outcome choice
    #                     bef_ephys['input'] = y.values
    #                     print()
    #                     print()
    #                     print(bef_ephys)
    #                     print(y)
    #                     bef_ephys.set_index('input', inplace=True, append=True)
                        
    #                     # the same for after the event
    #                     aft_ephys = smoothed.to_frame().copy()
    #                     aft_ephys['event_i'] = other_y_tpoints.get_indexer(aft_ephys.index)
    #                     # get the ephys after a single event and rest to count index (order after event)
    #                     aft_ephys = aft_ephys[aft_ephys['event_i'] != -1].groupby('event_i').apply(lambda x: x.reset_index(drop=True))
    #                     # then use this order to groupby the single timepoints after the event
    #                     aft_ephys = aft_ephys.drop(columns='event_i').unstack()
    #                     # add the input info, like cue outcome choice
    #                     print(aft_ephys)
    #                     print(y)
    #                     aft_ephys['input'] = y.values
    #                     aft_ephys.set_index('input', inplace=True, append=True)
                        
    #                     axs[row, 1].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                        
                        
    #                     if encodes_col in ['reward_sound', 'reward_valve', 'cue_entry']:
    #                         if encodes_col != 'cue_entry':
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='|', s=1000, alpha=0.6, color='green')
    #                         else:
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='|', s=1000, alpha=0.6, color='gray')
                                
                            
    #                         axs[row, 1].plot(np.arange(-bef_ephys.shape[1],0), bef_ephys.values.mean(axis=0), alpha=0.9, color='gray', label='Mean Before')
    #                         axs[row, 1].plot(np.arange(aft_ephys.shape[1]), aft_ephys.values.mean(axis=0), alpha=0.9, color='green', label='Mean After')

    #                         # Add shaded area for std
    #                         bef_mean = bef_ephys.values.mean(axis=0)
    #                         bef_std = bef_ephys.values.std(axis=0)/2
    #                         bef_x = np.arange(-bef_ephys.shape[1],0)
    #                         axs[row, 1].fill_between(bef_x, bef_mean-bef_std, bef_mean+bef_std, color='gray', alpha=0.2, label='Std Before')

    #                         aft_mean = aft_ephys.values.mean(axis=0)
    #                         aft_std = aft_ephys.values.std(axis=0)/2
    #                         aft_x = np.arange(aft_ephys.shape[1])
    #                         axs[row, 1].fill_between(aft_x, aft_mean-aft_std, aft_mean+aft_std, color='green', alpha=0.2, label='Std After')
                            
    #                         all_vals = np.concatenate([aft_mean-aft_std, aft_mean+aft_std,
    #                                                    bef_mean-bef_std, bef_mean+bef_std])
    #                         min_here, max_here = all_vals.min(), all_vals.max()
    #                         if min_here < ymin:
    #                             ymin = min_here
    #                         if max_here > ymax:
    #                             ymax = max_here

    #                     elif encodes_col in ['cue1_vs_cue2', 'cue1_vs_cue2_post', 'cue1_vs_cue2_post_wins', 'cue1_vs_cue2_post_losses']:
    #                         axs[row, 0].scatter(y.index, np.zeros_like(y), marker='|', s=1000, alpha=0.6, color=['purple' if v == 1 else 'orange' for v in y.values])
    #                         if encodes_col.endswith('_wins'):
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='.', s=100, alpha=0.6, color='green')
    #                         elif encodes_col.endswith('_losses'):
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='.', s=100, alpha=0.6, color='red')
    #                         # merge the before and after ephys data
    #                         both_ephys = pd.concat([bef_ephys, aft_ephys], axis=1)
    #                         if both_ephys.empty or y.unique().size < 2:
    #                             print("missing one cue")
    #                             continue
    #                         x = np.arange(-both_ephys.shape[1]//2, both_ephys.shape[1]//2)
    #                         cue1_mean = both_ephys.loc[(slice(None), 1),:].mean(axis=0)
    #                         cue2_mean = both_ephys.loc[(slice(None), 2),:].mean(axis=0)
    #                         axs[row, 1].plot(x, cue1_mean, alpha=0.9, color='purple', label=f'n={both_ephys.loc[(slice(None), 1),:].shape[0]} cue1')
    #                         axs[row, 1].plot(x, cue2_mean, alpha=0.9, color='orange', label=f'n={both_ephys.loc[(slice(None), 2),:].shape[0]} cue2')
    #                         axs[row, 1].legend(loc='best', fontsize=6)

    #                         # in between std part
    #                         cue1_std = both_ephys.loc[(slice(None), 1),:].std(axis=0)/2
    #                         axs[row, 1].fill_between(x, cue1_mean-cue1_std, cue1_mean+cue1_std, color='purple', alpha=0.2)

    #                         cue2_std = both_ephys.loc[(slice(None), 2),:].std(axis=0)/2
    #                         axs[row, 1].fill_between(x, cue2_mean-cue2_std, cue2_mean+cue2_std, color='orange', alpha=0.2)

    #                         all_vals = np.concatenate([cue1_mean-cue1_std, cue2_mean+cue2_std,])
    #                         min_here, max_here = all_vals.min(), all_vals.max()
    #                         if min_here < ymin:
    #                             ymin = min_here
    #                         if max_here > ymax:
    #                             ymax = max_here

    #                     elif encodes_col in ['R1choice_entry', 'R2choice_entry', 'R1choice_entry_wins', 'R1choice_entry_losses', 'R2choice_entry_wins', 'R2choice_entry_losses']:
    #                         axs[row, 0].scatter(y.index, np.zeros_like(y), marker='|', s=1000, alpha=0.8, color=['black' if v == 1 else 'gray' for v in y.values])
    #                         if encodes_col.endswith('_wins'):
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='.', s=100, alpha=0.6, color='green')
    #                         elif encodes_col.endswith('_losses'):
    #                             axs[row, 0].scatter(y.index, np.zeros_like(y), marker='.', s=100, alpha=0.6, color='red')
    #                         # merge the before and after ephys data
    #                         both_ephys = pd.concat([bef_ephys, aft_ephys], axis=1)
    #                         if y.unique().size < 2:
    #                             continue
                            
    #                         x = np.arange(-both_ephys.shape[1]//2, both_ephys.shape[1]//2)
    #                         stopped_mean = both_ephys.loc[(slice(None), 1),:].mean(axis=0)
    #                         not_stopped_mean = both_ephys.loc[(slice(None), 0),:].mean(axis=0)
    #                         axs[row, 1].plot(x, stopped_mean, alpha=0.9, color='black', label=f'n={both_ephys.loc[(slice(None), 1),:].shape[0]} stopped')
    #                         axs[row, 1].plot(x, not_stopped_mean, alpha=1, color='gray', label=f'n={both_ephys.loc[(slice(None), 0),:].shape[0]} skipped')
    #                         axs[row, 1].legend(loc='best', fontsize=6)

    #                         # in between std part
    #                         stopped_std = both_ephys.loc[(slice(None), 1),:].std(axis=0)/4
    #                         axs[row, 1].fill_between(x, stopped_mean-stopped_std, stopped_mean+stopped_std, color='black', alpha=0.2)
    #                         not_stopped_std = both_ephys.loc[(slice(None), 0),:].std(axis=0)/4
    #                         axs[row, 1].fill_between(x, not_stopped_mean-not_stopped_std, not_stopped_mean+not_stopped_std, color='gray', alpha=0.2)

    #                         all_vals = np.concatenate([stopped_mean-stopped_std, not_stopped_mean+not_stopped_std,])
    #                         min_here, max_here = all_vals.min(), all_vals.max()
    #                         if min_here < ymin:
    #                             ymin = min_here
    #                         if max_here > ymax:
    #                             ymax = max_here

    #                     for ax in axs[:, 2:].flatten():
    #                         ax.set_ylim(ymin, ymax)

    #                     # labels
    #                     axs[row, 1].set_xticks(np.arange(-bef_ephys.shape[1], aft_ephys.shape[1]+1, 4))
    #                     axs[row, 1].set_xticklabels(np.arange(-bef_ephys.shape[1], aft_ephys.shape[1]+1, 4) * bin_size_us / 1_000_000, 
    #                                                 rotation=45, ha='right', fontsize=8)
    #                     axs[row, 1].set_xlabel('t=0')

    #                 row += 1
    #                 continue






                    
    #                 #kinematics:
    #                 # y = y.dropna()
    #                 # x = sess_ensemble_proj[ensemble] #.iloc[y.index]

    #                 # print(x)
    #                 axs[row, 0].plot(y.index, y.values, '-', markersize=1, alpha=0.2)
    #                 axs[row, 0].set_title(f'{encodes_col}', fontsize=10, loc='left')

    #                 one_y = y[y >= y.quantile(0.8)]      # upper 20% values
    #                 other_y = y[y <= y.quantile(0.2)]    # lower 20% values
    #                 axs[row, 0].plot(one_y.index, one_y.values, '.', markersize=1, alpha=0.5, color='red')
    #                 axs[row, 0].plot(other_y.index, other_y.values, '.', markersize=1, alpha=0.5, color='blue')

    #                 # axs[i, 1].hist(y.values, bins=40, orientation='horizontal',)
    #                 # axs[i, 1].axhline(y.quantile(0.8), color='red', linestyle='--', label='80% quantile')
    #                 # axs[i, 1].axhline(y.quantile(0.2), color='blue', linestyle='--', label='20% quantile')
    #                 # #set log x axis
                    
    #                 # axs[i, 1].scatter(sess_ensemble_proj[ensemble], y, s=1, alpha=0.3)
    #                 ephys_one_y = sess_ensemble_proj.loc[one_y.index, ensemble]
    #                 ephys_other_y = sess_ensemble_proj.loc[other_y.index, ensemble]
    #                 axs[row, 1].scatter(ephys_one_y, one_y.values, s=1, alpha=0.3, color='red',)
    #                 axs[row, 1].axvline(ephys_one_y.mean(), color='red', linestyle='--')
    #                 axs[row, 1].scatter(ephys_other_y, other_y.values, s=1, alpha=0.3, color='blue',)
    #                 axs[row, 1].axvline(ephys_other_y.mean(), color='blue', linestyle='--')
    #                 axs[i, 1].set_xscale('log')
                    
    #                 pvalue = ttest_ind(ephys_one_y, ephys_other_y, equal_var=False).pvalue
    #                 axs[row, 1].set_title(f'* - p={pvalue:.5f}' if pvalue < 0.05 else '', fontsize=10)

    #                 # if x.isna().all() or y.isna().all():
    #                 #     continue
    #                 row += 1
                    
    #             plt.tight_layout()
    #             plt.savefig('quickviz.png', dpi=300)
    #             plt.savefig('outputs/ens_enc/ensemble_encodings_latents_S{}_A{}.svg'.format(s_id, ensemble))
    #             # plt.savefig('outputs/ens_enc/ensemble_encodings_kinematics_S{}_A{}.png'.format(s_id, ensemble))
    #             # time.sleep(20)
                

    #             # exit()





    #                 # model = LinearRegression()
    #                 # model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    #                 # # y_pred = model.predict(x.values.reshape(-1, 1))
    #                 # r2 = model.score(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    #                 # # print(f"Ensemble {ensemble} - {encodes_col} R^2: {r2:.3f}")

                    
    #                 # # see if outliers encode high versus low y metric
    #                 # high_x_mask = (x >x.std()).values
    #                 # low_x_mask = (x < -x.std()).values
    #                 # high_y = y[high_x_mask]
    #                 # low_y = y[low_x_mask]
                    
    #                 # _, p_val = ttest_ind(high_y, low_y, equal_var=False)
    #                 # n_low = low_x_mask.sum()
    #                 # n_high = high_x_mask.sum()
    #                 # if n_low < 50 or n_high < 50:
    #                 #     print(f"{n_low:_},{n_high:_},skipping{encodes_col}", end='   ')
    #                 #     continue

    #                 # aggr_results.append(pd.Series({ #"x": x.tolist(), "y": y.tolist(), 
    #                 # "n_low": n_low, "n_high": n_high, "r2": 
    #                 #  r2, "p_val_highlow_activation": p_val, 'avg_activation': x.mean(),
    #                 # }, name=(s_id, block_t, ensemble, encodes_col)))


    #                 # if ensemble == 'Assembly003':

    #                 #     axs[i].set_xlim(-4.2,4.2)
    #                 #     axs[i].scatter(np.clip(x[high_x_mask], -4,4), y[high_x_mask], s=1, alpha=0.3, color='red', )
    #                 #     axs[i].scatter(np.clip(x[low_x_mask], -4,4), y[low_x_mask], s=1, alpha=0.3, color='blue', )

    #                 #     # draw boxplot of high and low x values, without outlier points
    #                 #     axs[i].boxplot([y[high_x_mask], y[low_x_mask]],
    #                 #                 positions=[1, -1], widths=0.5, 
    #                 #                 # labels=['High X', 'Low X'], 
    #                 #                     showfliers=False,
    #                 #                 patch_artist=True, notch=True)
    #                 #     axs[i].hlines(y[high_x_mask].mean(), -4, 4, colors='red', linestyles='dashed')
    #                 #     axs[i].hlines(y[low_x_mask].mean(), -4, 4, colors='blue', linestyles='dashed',
    #                 #                 label='** p = {:.3f}'.format(p_val) if p_val < 0.05 else '')
    #                 #     axs[i].set_title(f'{encodes_col}, n={len(y):,}')
    #                 #     axs[i].tick_params(axis='both', which='major', labelsize=6)
    #                 #     # plt.legend(fontsize=12)
    #                 #     if p_val < 0.05:
    #                 #         axs[i].legend(fontsize=12, edgecolor='black',)

    #                 # Plot regression line (sorted for clarity)
    #                 # sort_idx = np.argsort(x)
    #                 # ax.plot(x.iloc[sort_idx], y_pred[sort_idx], color='red', linewidth=2, label='Regression line')

    #                 # ax.set_xlabel(f'{ensemble}', fontsize=8)
    #                 # ax.set_ylabel(encodes_col)
    #                 # ax.set_title(f'R^2: {r2:.3f}')
    #                 # ax.legend(fontsize=6)
    #         # if ensemble == 'Assembly003':
                
    #     # plt.tight_layout()
    #     # plt.show()    
        
    #     # if s_id == 3:
    #     #     aggr_results = pd.concat(aggr_results, axis=1).T
    #     #     print(aggr_results)
    #     #     aggr_results.index.rename(('session_id', 'ephys_timestamp', 'ensemble', 'behavior_variable'), inplace=True)
    #     #     print(aggr_results)
    #     #     print(aggr_results.iloc[20])
    #     #     aggr_results = aggr_results.reset_index()
    
    
    # aggr_results = pd.concat(aggr_results, axis=1).T
    # print(aggr_results)
    # aggr_results.index.rename(('session_id', 'ephys_timestamp', 'ensemble', 'behavior_variable'), inplace=True)
    # print(aggr_results)
    # print(aggr_results.iloc[20])
    # aggr_results = aggr_results.reset_index()
    # return aggr_results

    
    # # # Assign each behavior row to a bin
    # # behavior = behavior.copy()
    # # behavior['ephys_bin'] = ensemble_proj.index.get_indexer(behavior.index)
    # # behavior_grouped = behavior.groupby('ephys_bin').mean().round(3)
    # # # Only keep bins that exist in ensemble_proj
    # # behavior_grouped = behavior_grouped.loc[behavior_grouped.index >= 0]
    # # # Optionally select kinematic variables
    # # if kinematic_vars is None:
    # #     kinematic_vars = [col for col in behavior_grouped.columns if col.startswith('frame_')]
    # # # Prepare result DataFrame
    # # r2_results = pd.DataFrame(index=ensemble_proj.columns, columns=kinematic_vars, dtype=float)
    # # model = LinearRegression()
    # # # For each ensemble
    # # for ens in ensemble_proj.columns:
    # #     y = ensemble_proj[ens].iloc[behavior_grouped.index]
    # #     for regre in kinematic_vars:
    # #         x = behavior_grouped[regre]
    # #         # Fit regression
    # #         model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    # #         r2 = model.score(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    # #         r2_results.loc[ens, regre] = r2
    # #         if verbose and r2 > 0.03:
    # #             print(f"Ensemble {ens} - {regre} R^2: {r2:.3f}")
    # # return r2_results




























def get_ConcatenatedPCA40ms(fr_hz):
    # fill 0 firing rate neurons with the minimum firing rate of the session (0 Hz)
    # print(fr_z)
    # print(fr_z.isna().sum().sum()   )
    # fr_z = fr_z.groupby(level='session_id').apply(lambda x: x.fillna(x.min().min()))
    # fr_z = fr_z.apply(lambda unit_fr: unit_fr.fillna(unit_fr.min()))
    
    fr_hz = fr_hz.set_index(["from_ephys_timestamp","to_ephys_timestamp"], append=True)
    fr_hz.reset_index(level=(0,1,2,4), inplace=True, drop=True)

    fr_z = fr_hz.apply(lambda unit_fr: ((unit_fr - unit_fr.mean()) / unit_fr.std()))

    # print(f"Firing rate shape: {fr_z}")
    # print(fr_z)
    # exit()
    # print(fr_z.isna().sum().sum()   )
    # exit()
    
    Z = fr_z.values.T  # transpose to have neurons in rows, time bins in columns
    
    corr_matrix = np.corrcoef(Z)
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    eigenval_sum = np.sum(eigenvals)
    expl_var = eigenvals / eigenval_sum
    
    # calculate projection of the data onto the eigenvectors
    Z_pca =  eigenvecs @ Z
    PC_embeddings = pd.DataFrame(Z_pca.T, index=fr_z.index,
                                 columns=[f"PC{i+1}" for i in range(Z_pca.shape[0])])
    print(PC_embeddings)
    
    for i in range(PC_embeddings.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.hist(np.clip(PC_embeddings.iloc[:, i], a_min=-5, a_max=5), bins=100, alpha=0.5, label=f'PC{i+1}')
        plt.xlabel(f'PC{i+1}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of PC{i+1}')
        plt.legend()
        plt.show()

    PCs = pd.DataFrame(eigenvecs, columns=[f"PC{i+1}" for i in range(eigenvecs.shape[1])])
    PCs = pd.concat([PCs, pd.Series(eigenvals, name='eigenvalues')], axis=1)
    PCs = pd.concat([PCs, pd.Series(expl_var, name='explained_variance')], axis=1)
    return PCs

from numba import jit, prange
from matplotlib.gridspec import GridSpec
import itertools

@jit(nopython=True, parallel=True)
def _compute_assembly_activity_numba(assembly_templates, fr_data):
    n_assemblies, n_bins, n_neurons = assembly_templates.shape[1], fr_data.shape[0], fr_data.shape[1]
    assembly_activity = np.zeros((n_assemblies, n_bins))
    
    # Ensure contiguous arrays for better performance
    assembly_templates = np.ascontiguousarray(assembly_templates)
    fr_data = np.ascontiguousarray(fr_data)
    
    for assembly_idx in prange(n_assemblies):
        template = assembly_templates[:, assembly_idx].copy()  # Make contiguous copy
        projector = np.outer(template, template)
        np.fill_diagonal(projector, 0.0)
        
        # Make projector contiguous
        projector = np.ascontiguousarray(projector)
        
        for t in range(n_bins):
            spike_vector = fr_data[t, :].copy()  # Make contiguous copy
            assembly_activity[assembly_idx, t] = spike_vector.T @ projector @ spike_vector
    
    return assembly_activity

def get_ConcatenatedEnsambles40ms(PCs, all_fr_hz):
    """
        Estimate assemblies and their activity from 40 ms binned firing rates.

        The function:
        1) Determines the number of assemblies from PCA eigenvalues using the
        Marcenko-Pastur upper bound.
        2) Computes assembly templates with FastICA in the PCA subspace.
        3) Projects z-scored firing rates onto the templates to obtain activity.

        Parameters:
        PCs : pandas.DataFrame
            PC loading matrix of shape (n_neurons, n_pcs). Must contain columns
            'eigenvalues' and 'explained_variance', which are removed in place.
            Index identifies neurons.
        all_fr_hz : pandas.DataFrame
            Firing rates (n_bins x n_neurons) at 40 ms. Row index is a MultiIndex
            with level 'session_id'. Columns 'from_ephys_timestamp' and
            'to_ephys_timestamp' are removed to build the output index.

        Returns:
        assembly_templates : pandas.DataFrame
            Neuron x assembly weights, columns 'Assembly001', …; index matches PCs.
        assembly_activity : pandas.DataFrame
            Assembly activity with index on
            ('session_id','from_ephys_timestamp','to_ephys_timestamp') and one
            column per assembly; returned with a reset index.

        Raises:
        KeyError
            Missing required columns or 'session_id' level.
        ValueError
            ICA non-convergence.
    """
     
    eigenvalues = PCs.pop('eigenvalues')
    explained_variance = PCs.pop('explained_variance')
    print(PCs)
    
    from_ephys_timestamp = all_fr_hz.pop('from_ephys_timestamp',)
    to_ephys_timestamp = all_fr_hz.pop('to_ephys_timestamp',)
    session_id = all_fr_hz.index.get_level_values('session_id')
    # fill NaN z scores where no spike is detected in a session with the min value across all sessions
    # all_fr_z = all_fr_z.apply(lambda unit_fr: unit_fr.fillna(unit_fr.min()))
    all_fr_z = all_fr_hz.apply(lambda unit_fr: ((unit_fr - unit_fr.mean()) / unit_fr.std()))


    n_bins = all_fr_z.shape[0]
    n_neurons = PCs.shape[0]
    # Determine statistical threshold
    q = n_bins / n_neurons
    # Using Marcenko-Pastur distribution for threshold
    lambda_max = (1 + np.sqrt(1/q))**2
    
    # Count significant assemblies
    n_assemblies = np.sum(eigenvalues > lambda_max)
    print(f"Number of assemblies detected: {n_assemblies}")
    
    # Use FastICA on z-scored data to find assemblies
    print(f"Calculating assembly templates using FastICA with {n_assemblies} components...")
    ica = FastICA(n_components=n_assemblies, random_state=42, max_iter=500)
    ica.fit(all_fr_z@PCs.values[:, :n_assemblies])  # ICA expects samples x features
    assembly_templates = PCs.values[:, :n_assemblies]@ica.components_  # Transpose to neurons x assemblies
    
    print(f"Computing assembly activity for {n_assemblies} assemblies using JIT...")    
    assembly_activity = _compute_assembly_activity_numba(assembly_templates, all_fr_z.values)

    
    assembly_templates = pd.DataFrame(assembly_templates, index=PCs.index,
                                    columns=[f"Assembly{i+1:03d}" for i in range(n_assemblies)])
    
    idx = pd.MultiIndex.from_arrays([session_id,from_ephys_timestamp, to_ephys_timestamp],
                                    names=['session_id', 'from_ephys_timestamp', 'to_ephys_timestamp'])
    assembly_activity = pd.DataFrame(assembly_activity.T,
                                     columns=[f"Assembly{i+1:03d}" for i in range(n_assemblies)],
                                     index=idx)
    print(assembly_activity)

    session_stop_t = np.cumsum(assembly_activity.groupby('session_id').apply(lambda x: x.iloc[-1].name[1]))
    print(session_stop_t)
    
    for i in range(n_assemblies):
        plt.plot(np.cumsum(assembly_activity[f"Assembly{i+1:03d}"].index.get_level_values('from_ephys_timestamp')),
                 assembly_activity[f"Assembly{i+1:03d}"].values, 
                 label=f"Assembly {i+1:03d}")
        plt.vlines(session_stop_t, ymin=0, ymax=assembly_activity[f"Assembly{i+1:03d}"].max(),
                   colors='red', linestyles='dashed', label=f"Session {i+1:03d} stop")
        plt.xlabel('Ephys Timestamp')
        plt.show()
    return assembly_templates, assembly_activity.reset_index()


# def get_FiringRateTrackbinsHz(fr_z, track_behavior_data):
    
#     def trackwise_averages(fr, track_behavior_data):
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
#         return track_behavior_data.groupby('from_z_position_bin', ).apply(time_bin_avg)
    
#     # recover index
#     old_idx = fr_z.index
#     fr_z = fr_z.reset_index(drop=True)
#     new_idx = pd.IntervalIndex.from_breaks(
#         np.arange(0, fr_z.shape[0]*40_000+1, 40_000, dtype='uint64'),
#     )
#     fr_z.index = new_idx
#     fr_hz_averages = trackwise_averages(fr_z, track_behavior_data)
#     return fr_hz_averages
#     import matplotlib.pyplot as plt
#     plt.imshow(fr_hz_averages.values.T, aspect='auto', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#     print(fr_hz_averages)

def get_FiringRateTrackwiseHz(fr, track_behavior_data):
    def time_bin_avg(posbin_data, ):
        pos_bin = posbin_data.from_position_bin.iloc[0]
        print(f"{pos_bin:04}", end='\r')
        
        # to_posbin_t = trial_data.posbin_from_ephys_timestamp
        from_posbin_t = posbin_data.posbin_from_ephys_timestamp.values
        to_posbin_t = posbin_data.posbin_to_ephys_timestamp.values
        interval = pd.IntervalIndex.from_arrays(
            from_posbin_t-40_000,
            to_posbin_t+40_000,
            closed='both',
        )
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
        posbin_trial_wise_fr = trial_fr.groupby('posbin_t_edges', observed=True).mean()
        # posbin_trial_wise_fr /= interval.length.values[trials_exist_mask, None] /1e6 # Values are already in Hz as we use the 40ms Hz as input data for the function 

        # add meta data, cue outcome, position bin, trial id        
        posbin_trial_wise_fr['cue'] = posbin_data[trials_exist_mask].cue.values
        posbin_trial_wise_fr['trial_outcome'] = posbin_data[trials_exist_mask].trial_outcome.values
        posbin_trial_wise_fr['choice_R1'] = posbin_data[trials_exist_mask].choice_R1.values
        posbin_trial_wise_fr['choice_R2'] = posbin_data[trials_exist_mask].choice_R2.values
        posbin_trial_wise_fr['bin_length'] = interval.length.values[trials_exist_mask]/1e6
        posbin_trial_wise_fr.index = posbin_data.trial_id.values[trials_exist_mask]
        return posbin_trial_wise_fr        

    # recover interval index
    fr.index = pd.IntervalIndex.from_arrays(fr.pop("from_ephys_timestamp"), 
                                            fr.pop("to_ephys_timestamp"))

    print("Track bin: ")
    fr_hz_averages = track_behavior_data.groupby(['from_position_bin']).apply(time_bin_avg)
    fr_hz_averages.index = fr_hz_averages.index.rename(['from_position_bin', 'trial_id'],)
    fr_hz_averages.reset_index(inplace=True, drop=False)
    return fr_hz_averages

# def get_PCsZonewise(fr, track_behavior_data):
#     L = Logger()
#     PCs_var_explained = .8
#     fr = fr.set_index(['trial_id', 'from_position_bin', 'cue', 'trial_outcome', 
#                        'choice_R1', 'choice_R2'], append=True, )
#     fr.index = fr.index.droplevel(3) # entry_id

#     fr.drop(columns=['bin_length', ], inplace=True)
#     fr.columns = fr.columns.astype(int)
#     fr = fr.reindex(columns=sorted(fr.columns))
    
#     track_behavior_data = track_behavior_data.set_index(['trial_id', 'from_position_bin', 'cue', 'trial_outcome', 
#                                        'choice_R1', 'choice_R2'], append=True, )
#     track_behavior_data.index = track_behavior_data.index.droplevel(3) # entry_id
    
#     zones = {
#         'beforeCueZone': (-168, -100),
#         'cueZone': (-80, 25),
#         'afterCueZone': (25, 50),
#         'reward1Zone': (50, 110),
#         'betweenRewardsZone': (110, 170),
#         'reward2Zone': (170, 230),
#         'postRewardZone': (230, 260),
#         'wholeTrack': (-168, 260),
#     }
    
#     aggr_embeds = []
#     aggr_subspace_basis = []

#     #TODO each sessions has differnt number of neurons
#     fr = fr.reindex(np.arange(1,78), axis=1).fillna(0)

#     for i in range(4):
#         # which modality to use for prediction
#         if i == 0:
#             predictor = fr.iloc[:, :20]
#             predictor_name = 'HP'
#         elif i == 1:   
#             predictor = fr.iloc[:, 20:]
#             predictor_name = 'mPFC'
#         elif i == 2:
#             predictor = track_behavior_data.loc[:, ['posbin_velocity', 'posbin_acceleration', 'lick_count',
#                                            'posbin_raw', 'posbin_yaw', 'posbin_pitch']]
#             predictor_name = 'behavior'
#         elif i == 3:
#             predictor = fr.iloc[:]
#             predictor_name = 'HP-mPFC'
            
#         debug_msg = (f"Embedding {predictor_name} with PCA:\n")
#         for zone, (from_z, to_z) in zones.items():
#             zone_data = predictor.loc[pd.IndexSlice[:,:,:,:,np.arange(from_z,to_z)]].astype(float)
            
#             # Standardize features
#             scaler = StandardScaler()
#             zone_data.loc[:,:] = scaler.fit_transform(zone_data)
             
#             zone_data_Z_trialwise = zone_data.unstack(level='from_position_bin')
#             zone_data_Z_trialwise.index = zone_data_Z_trialwise.index.droplevel([0,1,2])
#             zone_data_Z_trialwise = zone_data_Z_trialwise.fillna(0).astype(float)
            
#             pca = PCA(n_components=PCs_var_explained)
#             embedded = pca.fit_transform(zone_data_Z_trialwise)
            
#             debug_msg += (f"{embedded.shape[1]:3d} PCs capture {pca.explained_variance_ratio_.sum():.2f} "
#                           f" of variance of {zone_data_Z_trialwise.shape[1]:5d} "
#                           f"input dims in {embedded.shape[0]:3d} trials at "
#                           f"{zone} ({from_z}, {to_z})\n")

#             embedded = np.round(embedded, 3)
#             aggr_embeds.append(pd.Series(embedded.tolist(), index=zone_data_Z_trialwise.index,
#                                          name=(zone, predictor_name)))
            
#             columns = pd.MultiIndex.from_product([[zone], [predictor_name], 
#                                                   np.arange(embedded.shape[1])])
#             aggr_subspace_basis.append(pd.DataFrame(pca.components_.round(3).T,
#                                                     columns=columns))
#         L.logger.debug(debug_msg)

#     aggr_embeds = pd.concat(aggr_embeds, axis=1).stack(level=0, future_stack=True)
#     aggr_embeds.reset_index(inplace=True, drop=False)
#     aggr_embeds.rename(columns={"level_5": "track_zone"}, inplace=True)
    
#     # the subspace basis is a 2D array with pricipal components as columns, rows
#     # indicate predictors and track zones
#     #TODO warning right now due to old stacking behavior, will be fixed in future
#     aggr_subspace_basis = pd.concat(aggr_subspace_basis, axis=0).stack(level=(0,1), future_stack=False)
#     aggr_subspace_basis.reset_index(inplace=True, drop=False)
#     aggr_subspace_basis.rename(columns={"level_1": "track_zone",
#                                         "level_2": "predictor_name",}, inplace=True)
#     aggr_subspace_basis.drop(columns='level_0', inplace=True)
    
#     # check for in proper sessions that lack spikes for most of the session
#     valid_ephys_mask = aggr_embeds['mPFC'].notna()
#     valid_beh_mask = aggr_embeds['behavior'].notna()
#     if not all(valid_ephys_mask == valid_beh_mask):
#         msg = (f"Ephys and behavior trials mismatch!\n"
#                f"Valid trials in ephys: {aggr_embeds.trial_id[valid_ephys_mask].unique()}, "
#                f"valid trials in behavior: {aggr_embeds.trial_id[valid_beh_mask].unique()} "
#                f"Using common set of trials")
#         Logger().logger.warning(msg)
#         aggr_embeds.dropna(how='any', inplace=True)
    
#     return aggr_subspace_basis, aggr_embeds

# def get_PCsSubspaceAngles(session_subspace_basis, all_subspace_basis):
#     session_subspace_basis.index = session_subspace_basis.index.droplevel((0,1))  # Drop the first level of the index if it exists
#     session_subspace_basis.set_index(["track_zone", 'predictor_name'], inplace=True, append=True)
#     session_subspace_basis.index = session_subspace_basis.index.droplevel("entry_id")  # Drop the entry_id level if it exists
    
#     all_subspace_basis.index = all_subspace_basis.index.droplevel((0,1))  # Drop the first level of the index if it exists
#     all_subspace_basis.set_index(["track_zone", 'predictor_name'], inplace=True, append=True)
#     all_subspace_basis.index = all_subspace_basis.index.droplevel("entry_id")  # Drop the entry_id level if it exists
#     # print(all_subspace_basis)
    
#     angle_aggr = []
#     comps_aggr = []
#     for zone in all_subspace_basis.index.unique(level='track_zone'):
#         print(zone, end='\r')
#         for predictor in all_subspace_basis.index.unique(level='predictor_name'):
#             # get the subspace basis for the session
#             # track_session_subspace = session_subspace_basis.loc[(zone, predictor), :]
#             # get the subspace basis for all sessions
#             zone_all_subspace = all_subspace_basis.loc[(slice(None),zone, predictor), :]
#             session_n_PCs = zone_all_subspace.groupby(level='session_id').apply(
#                                                        lambda x: x.notna().all(0).sum())
#             # some sessions have very low number of PCs, exclude those, will be NaN in the end
#             std = np.std(session_n_PCs.values)
#             cutoff = np.mean(session_n_PCs.values) - 2*std
            
#             too_few_PCs_mask = session_n_PCs < cutoff
#             n_PCs = session_n_PCs[~too_few_PCs_mask].min()
            
#             # # print(track_all_subspace)
#             # print(zone, predictor,)
#             # smth = session_n_PCs.to_frame(name='n_PCs').copy()
#             # smth['keep'] = smth.n_PCs > cutoff
#             # print(smth)
#             # print('-----')
            
#             # print(zone, predictor,)
#             # print(n_PCs)
#             # n_PCs = 200
            
#             # zone_all_subspace = zone_all_subspace.iloc[:, :n_PCs]
#             # zone_session_subspace = , :].iloc[:, :n_PCs]
            
#             # print(zone_all_subspace, zone_session_subspace)
#             s0_subspace = session_subspace_basis.loc[(slice(None), zone, predictor)].values[:, :n_PCs]
#             if np.isnan(s0_subspace[:, :n_PCs]).any() or s0_subspace.shape[1] < n_PCs:
#                 Logger().logger.warning(f"Session subspace has too few PCs for {zone} {predictor}.")
#                 continue
            
#             for s_id in session_n_PCs[~too_few_PCs_mask].index:
#                 s_compare_subspace = zone_all_subspace.loc[s_id, :].values[:, :n_PCs]
#                 # print(zone_all_subspace.loc[s_id, :])
                
#                 # print(s0_subspace, s_compare_subspace)
#                 # print(s0_subspace.shape, s_compare_subspace.shape)
                
#                 # print("--")
#                 M = s0_subspace.T @ s_compare_subspace
#                 s0_c, S, s_comp_h_c = np.linalg.svd(M)
                

#                 canonc_angles = np.arccos(np.clip(S, -1, 1))
#                 s0_c_subspace = (s0_subspace @ s0_c).astype(np.float16)
#                 # print(s0_c_subspace)
#                 # print(s0_c_subspace.shape)
#                 # CC_U = (s1_PCs_U @ U_c[:, :])
#                 # print(c1_U)
#                 # print(c1_U.shape)
                
#                 # canonc_angles = subspace_angles(s0_subspace[:, :n_PCs],
#                 #                                 s_compare_subspace[:, :n_PCs])
#                 # canonc_angles = np.rad2deg(canonc_angles)
#                 # print(canonc_angles)
                
#                 angle_aggr.append(pd.Series(canonc_angles, index=np.arange(n_PCs),
#                                             name=(zone, predictor, s_id)))
#                 comps_aggr.append(pd.DataFrame(s0_c_subspace, columns=pd.MultiIndex.from_product(
#                                                  [[zone], [predictor], [s_id], np.arange(n_PCs)],
#                                                  names=['track_zone', 'predictor', 'comp_session_id', 'CC_i'],
#                                              )))
                
                
#                 print(angle_aggr[-1])
#     if len(angle_aggr) == 0:
#         return None
    
#     angle_aggr = pd.concat(angle_aggr, axis=1).T
#     angle_aggr.index.names = ['track_zone', 'predictor', 'comp_session_id']
    
#     comps_aggr = pd.concat(comps_aggr, axis=1)
#     comps_aggr = comps_aggr.T
#     return comps_aggr.reset_index(), angle_aggr.reset_index()
            
def get_SVMCueOutcomeChoicePred(fr_z, beh):
    def fit_SVMs_with_bootstrap(X, Y, name, kernel, t, n_iterations=200):
        print("---")
        # if lbl_counts.min() < 5:
        #     print(f"Not enough samples for {name}: ({lbl_counts})")
        #     return None
        
        predictions = []
        rng = np.random.default_rng(42)

        # if kernel != 'linear':
        #     continue
        Cs = [0.1, .5, 1, 5, 10,]
        accs, f1s = [], []

        aggr_predictions = np.ones((n_iterations, len(X)), dtype=int) * -1
        print(aggr_predictions.shape)
        for i in range(n_iterations):
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], Y[indices]
            # only a single class check:
            if len(np.unique(y_boot)) < 2:
                print("Skipping SVM fit due to single class in bootstrap sample")
                continue
            oob_mask = np.ones(len(X), dtype=bool)
            oob_mask[indices] = False
            X_oob, y_oob = X[oob_mask], Y[oob_mask]

            if len(y_oob) < 5 or len(np.unique(y_oob)) < 2:
                print("Skipping SVM fit due to insufficient or single class in OOB sample")
                continue
            
            lbl_counts = pd.Series(Y).value_counts()
            if lbl_counts.min() < 3:
                print(f"Not enough samples for {name}: ({lbl_counts})")
                continue
            
             # Define a pipeline combining a scaler with the SVC
            

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
        
        
        # # get the "average" prediction across bootstrap iterations
        # aggr_predictions = pd.DataFrame(aggr_predictions)
        # aggr_predictions[aggr_predictions == -1] = np.nan
        # pred = aggr_predictions.mode(axis=0).iloc[0].values
        
        # difference
        # f1_aggr = classification_report(Y, pred, output_dict=True, zero_division=0)['macro avg']['f1-score']
        
        # print(f"{kernel}: mean: {np.mean(f1s):.3f}")
        
        ci = np.percentile(f1s, [2.5, 97.5])
        print(f"mean F1 {np.mean(f1s):.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], len: {len(f1s)}")
        return {
            'name': name,
            't': t,
            'n': len(f1s),
            'n_positives': int(np.sum(Y)),
            'n_negatives': int(len(Y) - np.sum(Y)),
            'kernel': kernel,
            'f1_mean': float(np.mean(f1s)),
            'f1_ci_lower': float(ci[0]),
            'f1_ci_upper': float(ci[1]),
            'acc_mean': float(np.mean(accs)),
            'acc_ci_lower': float(np.percentile(accs, 2.5)),
            'acc_ci_upper': float(np.percentile(accs, 97.5)),
        }

        
    print("---")
        
                
                
                
    L = Logger()
    
    fr_z = fr_z.set_index(["from_ephys_timestamp","to_ephys_timestamp"], append=True)
    fr_z.reset_index(level=(0,1,2,3,), inplace=True, drop=True)
    print(fr_z)
    
    enter_R1_t = beh.loc[:, ['trackzone_reward2_enter_ephys_t', 'choice_R2', 'trial_id']]
    enter_R1_t = enter_R1_t[~enter_R1_t.duplicated() & (enter_R1_t.trial_id != -1)]
    enter_R1_t = enter_R1_t.reset_index(drop=True).set_index('trial_id')
    Y = enter_R1_t.choice_R2.values
    print(enter_R1_t)
    print(Y)
    
    ball_vel = beh.loc[:, ['frame_raw', 'frame_yaw', 'frame_pitch', 'frame_ephys_timestamp']]
    print(ball_vel)
    max_t = 10_000_001  # 10s
    t_step = 1_000_000  # 1s
    n_frames = 58 # 1 s of frames at 59.5 Hz
    n_fr_bins = 24 # 1 s of 40ms bins at 25 Hz
    
    X_beh = []
    X_fr = []
    for R1_t in enter_R1_t.trackzone_reward2_enter_ephys_t:
        to_t = R1_t
        prv_R1_t = 0
        
        all_intervals_beh_X = {}
        all_intervals_fr_X = {}
        print("-.... ", R1_t)
        for i, from_t in enumerate(np.arange(R1_t - t_step, R1_t - max_t, -t_step)):
            if from_t < prv_R1_t:
                print("overlapping with previous trial")
                all_intervals_beh_X[i] = None
                all_intervals_fr_X[i] = None
                continue
            
            interv_beh = ball_vel[(ball_vel.frame_ephys_timestamp >= from_t) &
                             (ball_vel.frame_ephys_timestamp < to_t)].iloc[-n_frames:, :].dropna()
            if interv_beh.shape[0] < n_frames: # partial data, skip
                print(f"Not enough behavior data around R1 entry at {R1_t} ({interv_beh.shape[0]} samples), skipping interval ", i)
                all_intervals_beh_X[i] = None
                all_intervals_fr_X[i] = None
                continue
            
            if interv_beh.shape != (n_frames, 4):
                print(f"Behavior shape mismatch: {interv_beh.shape}")
                all_intervals_beh_X[i] = None
                all_intervals_fr_X[i] = None
                continue

            interv_fr = fr_z[(fr_z.index.get_level_values('from_ephys_timestamp') >= from_t)
                        & (fr_z.index.get_level_values('to_ephys_timestamp') < to_t)
                       ].iloc[-n_fr_bins:, :]
            
            if interv_fr.shape != (n_fr_bins, fr_z.shape[1]):
                print(f"Firing rate shape mismatch: {interv_fr.shape}")
                all_intervals_beh_X[i] = None
                all_intervals_fr_X[i] = None
                continue
            

            all_intervals_beh_X[i] = interv_beh[['frame_raw', 'frame_yaw', 'frame_pitch']].values.flatten()
            all_intervals_fr_X[i] = interv_fr.values.flatten()
            # print(all_intervals_beh_X[i])
            # print(all_intervals_fr_X[i])

            to_t = from_t
    
        print(all_intervals_beh_X.keys())
        X_beh.append(all_intervals_beh_X)
        X_fr.append(all_intervals_fr_X)
    # print(X_beh)
    # print(X_fr)



    results = []
    for t in range(0, 10):
        print("\n\nTime before R1: ", t+1)
        
        invalid_trial_mask = []
        X_beh_t = []
        X_fr_t = []
        for trial_beh, trial_fr in zip(X_beh, X_fr):
            # [print(v, end=' ') if v is None else print("Ok", end=' ') for k,v in trial_beh.items()]
            if trial_beh[t] is None or trial_fr[t] is None:
                print("Skipping trial due to missing data")
                invalid_trial_mask.append(True)
            else:
                invalid_trial_mask.append(False)
                X_beh_t.append(trial_beh[t])
                X_fr_t.append(trial_fr[t])
        
        
        
        X_beh_t = np.stack(X_beh_t)
        X_fr_t = np.stack(X_fr_t)
        Y_t = Y[~np.array(invalid_trial_mask)]
        
        print(f"Using {X_beh_t.shape[0]} trials for SVM fitting")
        print(X_beh_t.shape, X_fr_t.shape, Y_t.shape)


        # pca on behavior
        pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
        X_beh_t_pca = pca.fit_transform(X_beh_t)
        print(f"Behavior PCA reduced to {X_beh_t_pca.shape[1]} dimensions from {X_beh_t.shape[1]} dimensions")
        
        X_fr_t = np.nan_to_num(X_fr_t, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # pca on firing rates
        pca = PCA(n_components=0.95, svd_solver='auto', random_state=42)
        X_fr_t_pca = pca.fit_transform(X_fr_t)
        print(f"Firing rate PCA reduced to {X_fr_t_pca.shape[1]} dimensions from {X_fr_t.shape[1]} dimensions")
        
        res1 = fit_SVMs_with_bootstrap(X_beh_t_pca, Y_t, f"RawYawPitch", kernel='linear', t=t+1)
        res2 = fit_SVMs_with_bootstrap(X_fr_t_pca, Y_t, f"FiringRate", kernel='linear', t=t+1)
        res3 = fit_SVMs_with_bootstrap(X_beh_t_pca, Y_t, f"RawYawPitch", kernel='rbf', t=t+1)
        res4 = fit_SVMs_with_bootstrap(X_fr_t_pca, Y_t, f"FiringRate", kernel='rbf', t=t+1)

        results.extend([res1, res2, res3, res4])

        print(results)
    return pd.DataFrame(results)



def get_PVCueCorr(trackfr_data, track_behavior_data):
    def compute_population_vector_correlation(PVs):
        # Split the data into cue 1 and cue 2
        pv1 = PVs.xs(1, level='cue')
        pv2 = PVs.xs(2, level='cue')
        unif_index = pv1.index.intersection(pv2.index)
        pv1 = pv1.loc[unif_index]
        pv2 = pv2.loc[unif_index]
        
        spatial_avg_pv1 = []
        spatial_avg_pv2 = []
        for i in range(0, len(unif_index), 5):
            spatial_bins = unif_index[i:i+5]
            spatial_avg_pv1.append(pv1.loc[spatial_bins].mean(0).rename(spatial_bins[0]))
            # print(spatial_avg_pv1)
            spatial_avg_pv2.append(pv2.loc[spatial_bins].mean(0).rename(spatial_bins[0]))
            # print(spatial_avg_pv2)
            # exit()
            
        spatial_avg_pv1 = pd.concat(spatial_avg_pv1, axis=1).T
        spatial_avg_pv2 = pd.concat(spatial_avg_pv2, axis=1).T

        # Calculate correlation matrix for averaged bins
        corr_matrix = np.zeros((len(spatial_avg_pv1), len(spatial_avg_pv2)))
        for i, cue1_spatial_bin in enumerate(spatial_avg_pv1.index):
            x = spatial_avg_pv1.loc[cue1_spatial_bin]
            # print(x)
            for j, cue2_spatial_bin in enumerate(spatial_avg_pv2.index):
                y = spatial_avg_pv2.loc[cue2_spatial_bin]
                # corr_matrix[(cue1_spatial_bin, cue2_spatial_bin)] = np.corrcoef(x, y)[0, 1]
                corr_matrix[i, j] = np.corrcoef(x, y)[0, 1]
        return pd.DataFrame(corr_matrix, 
                            index=spatial_avg_pv1.index, 
                            columns=spatial_avg_pv2.index)
    
    L = Logger()
    # TODO subset into trial_outcome group and choice and trial 1/3 2/3 3/3
    trackfr_data = trackfr_data.set_index(['trial_id', 'from_position_bin', 'cue', 'trial_outcome', 
                                           'choice_R1', 'choice_R2'], append=True)
    trackfr_data.index = trackfr_data.index.droplevel((0,1,2,3))
    trackfr_data.drop(columns=['bin_length'], inplace=True)
    fr = trackfr_data.reindex(np.arange(1,78).astype(str), axis=1).fillna(0)
     
    track_behavior_data = track_behavior_data.set_index(['trial_id', 'from_position_bin', 
                                                         'cue', 'trial_outcome', 
                                       'choice_R1', 'choice_R2'], append=True, )
    track_behavior_data.index = track_behavior_data.index.droplevel(3) # entry_id
    
    PVCueCorr_aggr = []
    for i in range(4):
        # which modality to use for prediction
        if i == 0:
            predictor = fr.iloc[:, :20]
            predictor_name = 'HP'
        elif i == 1:   
            predictor = fr.iloc[:, 20:]
            predictor_name = 'mPFC'
        elif i == 2:
            predictor = track_behavior_data.loc[:, ['posbin_velocity', 'posbin_acceleration', 'lick_count',
                                                    'posbin_raw', 'posbin_yaw', 'posbin_pitch']]
            predictor_name = 'behavior'
        elif i == 3:
            predictor = fr.iloc[:]
            predictor_name = 'HP-mPFC'
            
        debug_msg = (f"Correlating population vector: {predictor_name}\n")
        
        # mean over trials
        PVs = predictor.groupby(level=('from_position_bin', 'cue')).mean()
        cross_corr = compute_population_vector_correlation(PVs)
        print(cross_corr)
        cross_corr['predictor'] = predictor_name
        cross_corr = cross_corr.reset_index().rename(columns={'index': 'cue1_from_position_bin'})
        PVCueCorr_aggr.append(cross_corr)
            
        # if i == 0:
        #     # import matplotlib.pyplot as plt
        #     plt.figure(figsize=(10, 8))
        #     print(cross_corr)
        #     print(cross_corr.cue1_from_position_bin)
        #     idx = cross_corr.pop('cue1_from_position_bin')
        #     cross_corr.pop('predictor')
        #     plt.imshow(cross_corr, aspect='auto', interpolation='nearest', vmin=.5, vmax=1)
        #     plt.colorbar(label='Correlation')
        #     plt.title("Population Vector Correlation (5-bin averages)")
        #     plt.xlabel("Position Bin (Cue 1)")
        #     plt.ylabel("Position Bin (Cue 2)")
            
        #     # Add tick labels for averaged bins
        #     tick_positions = np.arange(len(cross_corr))
        #     tick_labels = [f"{x:.0f}" for x in idx]
        #     plt.xticks(tick_positions[::2], tick_labels[::2], rotation=45)
        #     plt.yticks(tick_positions[::2], tick_labels[::2])
            
        #     plt.tight_layout()
        #     plt.savefig("population_vector_correlation.png", dpi=300, bbox_inches='tight')
            # exit()
    PVCueCorr_aggr = pd.concat(PVCueCorr_aggr, axis=0)
    return PVCueCorr_aggr