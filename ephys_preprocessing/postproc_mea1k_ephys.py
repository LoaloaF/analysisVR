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
        
def get_Spikes(session_fullfname,):
    L = Logger()
    
    nas = device_paths()[0]
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    animal_name = session_name.split("_")[2]
    ss_dirnames = pd.read_csv(os.path.join(nas, "devices", "animal_meta_info.csv"), 
                              index_col='animal_name').loc[animal_name, 'ss_dirnames'].split(",")
    traces, mapping = ml.session_modality_from_nas(session_fullfname, key='ephys_traces')
    if mapping is None:
        return None
    

    session_spike_table = []
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
        
        # print(mapping)
        
        with h5py.File(ss_res_fullfname, 'r') as f:
            # wf_length = int(f['filtShape'][0][0])
            # l = len(spike_times)

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
                "cluster_id": (spike_clusters +(10_000*ss_i)).astype(int), # make ids unique
                "amplitude_uV": spike_amps,
                "channel": spike_sites-1,
                "shank": np.round(spike_positions)//1000,
            })
            spike_table.loc[spike_clusters==-1, 'cluster_id'] = -1 # reset MultiUnit back to -1
            L.logger.debug(f"Extracted spikes:\n{spike_table}")
            spike_table['shank_name'] = mapping.shank_name.values[spike_table.channel]
            spike_table['depth'] = mapping.depth.values[spike_table.channel]
            spike_table['fiber_id'] = mapping.el_pair.values[spike_table.channel]
            spike_table['pad_id'] = mapping.pad_id.values[spike_table.channel]
            spike_table['mea1k_el'] = mapping.mea1k_el.values[spike_table.channel]
            
            # cluster table
            cluster_table = pd.DataFrame({
                # "cluster_id": f['clusterIDs'][0],
                "clusterCenters": f['clusterCenters'][0],
                "unit_count": f['unitCount'][0],
                "unit_snr": f['unitSNR'][0],
                "unit_Vpp": f['unitVpp'][0],
                "unit_VppRaw": f['unitVppRaw'][0],
            })
            
            spike_table['waveform'] = [[np.int16(0)] * 20] * len(spike_table)
            session_spike_table.append(spike_table)
            
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
            
    session_spike_table = pd.concat(session_spike_table, axis=0)      
    
    
    unique_ids = session_spike_table.cluster_id.unique()
    renamer = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids))}
    renamer[-1] = -1 # reset MultiUnit to -1
    session_spike_table['cluster_id'] = session_spike_table['cluster_id'].replace(renamer)
    
    # assign colors per cluster    
    map_colors = np.vectorize(make_discr_cluster_id_cmap(session_spike_table['cluster_id']).get)
    session_spike_table['cluster_color'] = map_colors(session_spike_table['cluster_id'])

    return session_spike_table