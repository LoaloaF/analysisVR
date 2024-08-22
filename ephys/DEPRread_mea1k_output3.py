import os
import pickle
from glob import glob
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
    
MAX_AMPL_mV = 2900.
BITRATE = 2**10
GAIN = 7
SR = 20_000
SR = 20_000
NPULSES = 30

def read_hdf5_maxwell(full_rec_fname, precomputed='to', 
                      chnl_subset=slice(None), time_subset=slice(None)):
    npyfname = os.path.basename(full_rec_fname).replace(".raw.h5", "_stim_ttls.pkl")
    np_fullfname = os.path.join(os.path.dirname(full_rec_fname), npyfname)

    if precomputed != "from":
        with h5py.File(full_rec_fname, 'r') as file:
            print(f'Keys in H5 file: {np.array(file.keys())}')
            
            mapping = np.array([list(m)[:2] for m in file['mapping']])
            gain = np.array(file["settings/gain"])[0]
            global GAIN
            if gain != GAIN:
                GAIN = gain
            
            t = str(np.array(file['time'])[0])
            start_t = pd.Timestamp(t[t.find(": ")+2:t.find(";")])
            stop_t = pd.Timestamp(t[t.find("stop: ")+6:-3])
            
            stim_ttl = np.array(file['sig'][1024, :])
            
            dataset = np.array(file['sig'][chnl_subset, time_subset])
            dataset = convert_to_vol(dataset).astype(np.ushort)
            
        if precomputed == "to":
            np_fullfname = os.path.join("/home/loaloa/impedance/rec1/recordings", npyfname)
            np.save(np_fullfname, dataset)
        print((f"From {start_t} to {stop_t}, sampled at {SR}Hz, gain:{GAIN}:"
              f"\n{dataset}\n{dataset.shape}\nMapping:\n{mapping},{mapping.shape}"))
        return dataset, mapping, start_t, stop_t
    
    elif precomputed == "from":
        dataset = np.load(np_fullfname)
        return dataset, None, None, None
    
def get_n_stimsets(path):
    return len(glob(f"{path}/stim_set_*"))

def get_n_chunks_for_stimset(stim_set_dir):
    return len(glob(f"{stim_set_dir}/mxConfig_chunk*_map00.npy"))

def get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i):
    return len(glob(f"{stim_set_dir}/mxConfig_chunk{chunk_i}_map*.npy"))

def get_stimset_stim_events(stim_ttl, stim_set_dir, stim_set_i):
    diffdata = np.diff(stim_ttl)
    t = np.arange(diffdata.size)[diffdata==65525]

    nchunks = get_n_chunks_for_stimset(stim_set_dir)
    stim_set_i_ttls = {}
    ttl_i = 0
    for chunk_i in range(nchunks):
        n_elsets = get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i)
        for elset_i in range(n_elsets):
            for pulse_i in range(NPULSES):
                stim_set_i_ttls[(stim_set_i, chunk_i, elset_i, pulse_i)] = t[ttl_i]
                ttl_i += 1
    return stim_set_i_ttls    

def get_resolution():
    return (MAX_AMPL_mV/BITRATE) /GAIN

def get_max_ampl():
    res = get_resolution()
    return BITRATE/2 *res

def convert_to_vol(data):
    # scale data to mv
    res = get_resolution()
    data = (data-BITRATE/2) *res
    # floating/ offset
    data += MAX_AMPL_mV/2 
    return data

def read_mxconfig(path, chunk_i):
    mapping_fname = f'mxConfig_chunk{chunk_i}.cfg' if chunk_i != -1 else f'mxConfig.cfg'
    with open(os.path.join(path, mapping_fname), 'r') as file:
        mapping = file.readline().split(";")
    data = []
    for m in mapping[:-1]:
        # Split the string on parentheses and slash to get the four values
        channel, rest = m.split('(')
        electrode, rest = rest.split(')')
        xloc, yloc = rest.split('/')
        # Append the values as a list to the data list
        # data.append([int(channel), int(electrode), float(xloc), float(yloc)])
        data.append([int(channel), int(electrode)])
    return np.array(data)

def read_stim_map(path, chunk_i, j):
    stimmap_fname = f'mxConfig_chunk{chunk_i}_map{j:02d}.npy'
    return np.load(os.path.join(path, stimmap_fname), 'r').astype(int).T
    
def read_stim_tstamps(path, chunk_i, j):
    tstamps_fname = f'mxConfig_chunk{chunk_i}_tstamp{j:02d}.npy'
    tstamps = np.load(os.path.join(path, tstamps_fname))
    return tstamps

def check_electrode_mapping(data, mapping):
    print()
    non_mapped_data = [c for c in np.arange(data.shape[0]) if c not in mapping[:, 0]]
    print(f"{len(non_mapped_data)} rows in the data are not in the mapping: {non_mapped_data}\n{data[non_mapped_data]}")
    # data = np.delete(data, non_mapped_data, axis=0)

    print()
    chnl_cnts = np.unique(mapping[:, 0], return_counts=True)[1]
    dupl_chnls = np.unique(mapping[:, 0])[np.where(chnl_cnts > 1)]
    dupl_chnls_cnt = chnl_cnts[chnl_cnts > 1]
    print(f"Non-unique channels ({dupl_chnls.size}), Sum:{(chnl_cnts-1).sum()}:")
    for c in dupl_chnls:
        [print(m) for m in mapping[mapping[:, 0]==c]]
        print()
    
    print()
    electr_cnts = np.unique(mapping[:,1], return_counts=True)[1]
    dupl_electr = np.unique(mapping[:,1])[np.where(electr_cnts > 1)]
    dupl_electr_cnt = electr_cnts[electr_cnts > 1]
    print(f"Non-unique electrodes ({dupl_electr.size}), Sum:{(electr_cnts-1).sum()}:")
    for c in dupl_electr:
        [print(m) for m in mapping[mapping[:,1]==c]]
        print()

    return non_mapped_data, dupl_chnls, dupl_electr

def sort_by_electrode(data, mapping):
    # duplicates in mapping channels
    mapping = mapping[np.unique(mapping[:, 0], return_index=True)[1]]
    mapping = mapping[mapping[:, 0] <data.shape[0]]
    reindexer = np.argsort(mapping[:,1])
    return data[reindexer], mapping[reindexer]

def get_timesindex(start_time, sr, data):
    return pd.date_range(start_time, periods=data.shape[1], freq=f"{1/sr*1e9}ns")


def setup_singlelineplot(figsize=(13,7)):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    fig.subplots_adjust(top=.99, bottom=.1, right=.99, left=0.08)
    [ax.spines[w].set_visible(False) for w in ('top', 'right', 'left')]# 'bottom', "left")]
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def draw_singledata(ax, data, mapping, stim_mapping, channel_subset, dupl_electr, xindex, stim_tstamps=None):
    maxampl = get_max_ampl()/4
    ylims = MAX_AMPL_mV/2 -maxampl,  MAX_AMPL_mV/2 +maxampl
    yrange = (ylims[1]-ylims[0]) *1
    offsets = (np.arange(len(channel_subset)) *yrange).astype(np.ushort)
    data[channel_subset] += offsets[:,np.newaxis]
    # print("data ", data[channel_subset])
    
    ycenters = offsets + (2900/2)
    ymins = ycenters -yrange/2 *.9
    ymaxs = ycenters +yrange/2 *.9
    # print(ycenters)
    # print(ymins)
    # print(ymaxs)
    
    ax.set_yticks(ycenters, minor=False)
    ax.set_yticks(np.concatenate([ymins, ymaxs]), minor=True)
    lbls = []
    i = 0
    print()
    # print(mapping.shape)
    for data_i in range(data.shape[0]):
        if channel_subset is not None and data_i not in channel_subset:
            continue
        ax.plot(xindex, data[data_i, :], lw=.4)
        ax.vlines(xindex[0], ymins[i], ymaxs[i], color='k', lw=.4)
        ax.hlines(ycenters[i], xindex[0], xindex[-1], color='k', lw=.3, linestyle='dashed')
        m = mapping[mapping[:, 0] == data_i]
        if any([True for v in m[:,1] if v in dupl_electr]):
            ax.vlines(xindex[0], ymins[i], ymaxs[i], color='k', alpha=.3, lw=4)
        if any([True for v in m[:,1] if v in stim_mapping[:,1]]):
            ax.vlines(xindex[-6000], ymins[i], ymaxs[i], color='b', alpha=.3, lw=4)
        if data_i in stim_mapping[:,0]:
            ax.vlines(xindex[-1], ymins[i], ymaxs[i], color='g', alpha=.3, lw=4)
        
        
        for j in stim_tstamps:
            ax.vlines(xindex[j], ymins[i], ymaxs[i], color='g', alpha=.3, lw=1)

        lbls.append(m)
        # lbls.append(f'C:{int(mapping[:, 0][i]):04d} E:{int(mapping[:,1][i]):05d}')

        if i == 0:
            ax.text(xindex[0], ymins[0],f'{maxampl:.2f}mV', ha='right', va='center')
            ax.text(xindex[0], ymaxs[0], f'{-maxampl:.2f}mV', ha='right', va='center')
        i += 1
    ax.set_ylim(ymaxs[-1], ymins[0])
    ax.set_yticklabels(lbls)

    # for fr in [1, 5, 10, 30, 60]:
    #     xticklabels = pd.date_range(xindex[0], xindex[-1], freq=f"{fr}s")
    #     print(xticklabels)
    #     if (3< xticklabels.size <15):
    #         print("iN")
    #         print(xticklabels)
    #         ax.set_xticks(xticklabels)
    #         ax.set_xticklabels(xticklabels.strftime('%M:%S'))
    #         ax.set_xlim(xindex[0], xindex[-1])
    #         break

def read_stim_config(fullfname):
    stim_mapping = np.load(fullfname, 'r').astype(int).T
    print("Stimulation map: ", stim_mapping)
    return stim_mapping

def butter_highpass_filter(data, cutoff, fs, order=5):
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def subtract_median(data):
    med = np.median(data, axis=1)
    data = data -med[:, np.newaxis]
    return data

def extract_stim_events(rec_dir, recording_fname, precomputed=None, debug=True):
    # stim_set_i (25), chunk_i (4), elset_i (11-13), pulse_i (30)
    full_rec_fname = os.path.join(rec_dir, 'recordings', recording_fname)
    precomputed_fname = full_rec_fname.replace("raw.h5", "stim_ttls.pkl")
    if precomputed == "from":
        return pickle.load(open(precomputed_fname, 'rb'))
    
    with h5py.File(full_rec_fname, 'r') as file:
        stim_ttl = np.array(file['sig'][1024, :])

    diffdata = np.diff(stim_ttl)
    t = np.arange(diffdata.size)[diffdata==65511]
    # t = np.arange(diffdata.size)[diffdata==65525]

    n_stim_sets = get_n_stimsets(rec_dir)
    n_stim_sets = 1
    ttls = {}
    for stim_set_i in range(n_stim_sets):
        stim_set_dir = os.path.join(rec_dir, f"stim_set_{stim_set_i:02d}")

        nchunks = get_n_chunks_for_stimset(stim_set_dir)
        ttl_i = 0
        for chunk_i in range(nchunks):
            n_elsets = get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i)
            for elset_i in range(n_elsets):
                for pulse_i in range(NPULSES):
                    ttls[(stim_set_i, chunk_i, elset_i, pulse_i)] = t[ttl_i]
                    ttl_i += 2
    if precomputed == "to":
        # save the dict ttls to a file and pickle
        pickle.dump(ttls, open(precomputed_fname, 'wb'))
    
    if debug:
        plt.plot(diffdata)
        for k, v in ttls.items():
            plt.text(v, stim_ttl[v], f"{k}", fontsize=8)
        plt.show()
    return ttls


def quicktest(rec_dir, recording_fname, stim_ttls):
    full_rec_fname = os.path.join(rec_dir, 'recordings', recording_fname)
    file = h5py.File(full_rec_fname, 'r')

    #n_stim_sets = get_n_stimsets(rec_dir)
    stim_set_i = 1
    chunk_i = 0
    elset_i = 0
    #tstart = stim_ttls[stim_set_i, chunk_i, elset_i, 0]
    #tend = tstart+110*NPULSES

    t_context_length = 560000    
    lbl = [185,615,721,811,910,935]
    #lbl = [158,584, 877,997]
    elset_data = np.array(file['sig'][lbl, 0:file['sig'].shape[1]])
    #tstart-t_context_length
    t_index = np.arange(0,file['sig'].shape[1])
    plt.plot(t_index, elset_data.T, label=lbl)
    plt.legend()
    plt.show()

def extract_eletrode_pulses(rec_dir, recording_fname, stim_ttls, debug=True):
    def process_elset():
        # (32,2) mapping from channel (1028) to stim channel electrode (26400)
        channel_map = np.stack([mapping[(mapping[:,1] == stim_el)] for stim_el in stim_map[0]])[:,0]
        channel_map = channel_map[np.argsort(channel_map[:,0])]

        index = [(stim_set_i, chunk_i, elset_i, stim_el) for stim_el in channel_map[:,1]]
        index = pd.MultiIndex.from_tuples(index, names=["stim_set", "chunk", "elset", "el"])
        
        tstart = stim_ttls[stim_set_i, chunk_i, elset_i, 0]
        tend = tstart+110*NPULSES
        elset_data = np.array(file['sig'][channel_map[:,0], tstart:tend])

        # save data in stim_set_dir
        elset_data = pd.DataFrame(elset_data, index=index)
        elset_data.to_csv(os.path.join(stim_set_dir, f"impedance_elset_{elset_i:02d}.csv"))
        
        # debugging
        if debug:
            print("Channel map:\n", channel_map)
            print("tstamps: ", tstart, tend)
            t_context_length = 30000
            fromt = max(0, tstart-t_context_length)
            tot = min(file['sig'].shape[1], tstart+t_context_length)
            elset_data = np.array(file['sig'][channel_map[:,0], fromt:tot])
            
            t_index = np.arange(fromt, tot)
            plt.plot(t_index, elset_data.T, label=channel_map[:,0])
            #draw vertical lines at tstart and tend
            plt.axvline(x=tstart, color='k', alpha=.5, linestyle='--')
            plt.axvline(x=tend, color='k', alpha=.5, linestyle='--')
            plt.title(f"Stim set {stim_set_i}, chunk {chunk_i}, elset {elset_i} (n={channel_map.shape[0]})")
            plt.legend(ncol=4)
            plt.show()
            
    full_rec_fname = os.path.join(rec_dir, 'recordings', recording_fname)
    file = h5py.File(full_rec_fname, 'r')

    n_stim_sets = get_n_stimsets(rec_dir)
    for stim_set_i in range(n_stim_sets):
        stim_set_i += 1
        print(f"\n\n--------\nProcessing stimulation set {stim_set_i}---------")
        stim_set_dir = f"stim_set_{stim_set_i:02d}"
        stim_set_dir = os.path.join(rec_dir, stim_set_dir)
        print(get_n_chunks_for_stimset(stim_set_dir))
        for chunk_i in range(get_n_chunks_for_stimset(stim_set_dir)):
            print(f"\nchunk {chunk_i}:")
            # classic 1028 channels mapping to electrode (max 26400)
            mapping = read_mxconfig(stim_set_dir, -1)
            #mapping = read_mxconfig(os.path.join(rec_dir, "stim_set_00"), -1)
            print(mapping[:18])

            n_elsets = get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i)
            for elset_i in range(n_elsets):
                print(f"elset{elset_i}",end="...")
                # (32,2) mapping from electrode (26400) to stim channel (32)
                stim_map = read_stim_map(stim_set_dir, chunk_i, elset_i)
                process_elset()
                
    file.close()

def main():
    rec_dir = './impedance/rec3/'
    #rec_dir = '/mnt/NTnas/mea1k/impedance/rec1/'
    recoding_fname = 'Trace_20240308_18_17_21.raw.h5'
    recoding_fname = 'el_set_01.raw.h5'
    #stim_ttls = extract_stim_events(rec_dir, recoding_fname, precomputed='to', debug=False)
    #extract_eletrode_pulses(rec_dir, recoding_fname, stim_ttls, debug=True)
    quicktest(rec_dir, recoding_fname, stim_ttls=None)



		
    # TODO double array.download() ??
    # TODO wait after inital download
    # TODO wait after connecting stimulation units (beofre first chunk)
    # TODO check chunk specific configs 
    # TODO rm timestamps.npy
    # TODO INit doesn't init?













def mainold():
    rec_dir = '/mnt/NTnas/mea1k/impedance/rec1/'
    rec_dir = './impedance/rec2/'
    recoding_fname = 'Trace_20240308_19_32_01.raw.h5'
    full_rec_fname = os.path.join(rec_dir, 'recordings', recoding_fname)

    channel_subset, time_subset = slice(10), slice(30000)
    data, h5mapping, start_t, stop_t = read_hdf5_maxwell(full_rec_fname, 
                                                        precomputed="from", 
                                                        chnl_subset=channel_subset, 
                                                        time_subset=time_subset)
    start_t = "2024-03-08 18:17:23"
    stim_set_i = 0 # (25)
    chunk_i = 3 # (4)
    elset_i = 5 # (11-13)

    stim_set_dir = f"stim_set_{stim_set_i:02d}"
    stim_set_dir = os.path.join(rec_dir, stim_set_dir)
    # classic 1028 channels mapping to electrode (max 26400)
    mapping = read_mxconfig(stim_set_dir, chunk_i)
    print(mapping)

    # stim_set_i (25), chunk_i (4), elset_i (11-13), pulse_i (30)
    stim_events_map = get_stimset_stim_events(data[1024], stim_set_dir, stim_set_i)
    [print(o) for o in stim_events_map]
    
    # (32,2) mapping from electrode (26400) to stim channel (32)
    stim_map = read_stim_map(stim_set_dir, chunk_i, elset_i)
    print(stim_map)
    
    # check if all data channels are in the mapping 
    unm_data, dupl_chnls, dupl_electr = check_electrode_mapping(data, mapping)

    # data = subtract_median(data)
    # data = butter_highpass_filter(data, 300, 2e4)
    xindex = get_timesindex(start_t, SR, data)
    channel_subset = np.stack([mapping[(mapping[:,1] == stim_el)][0] for stim_el in stim_map[0]])[:,0]
    print(channel_subset)

    channel_subset = np.append(channel_subset, [1024])
    # channel_subset = np.append(channel_subset, unm_data)
    np.save("ttls.npy", data[1024])
    figsize = (4.5, 7)
    fig, ax = setup_singlelineplot(figsize)
    # channel_subset = mapping[mapping[]]
    stim_tstamps = [(stim_events_map[stim_set_i, chunk_i, elset_i, pulse_i]) for pulse_i in range(NPULSES)]
    draw_singledata(ax, data, mapping, stim_map, channel_subset, dupl_electr, xindex, stim_tstamps)
    plt.show()

    start = 0
    nrows = 20
    nfigs = 1
    channel_subset = [183,198]
    for endrow in range(start, start+nfigs*nrows, nrows):
         figsize = (4.5, 7)
         fig, ax = setup_singlelineplot(figsize)
         channel_subset = np.arange(endrow, endrow+nrows)
         print("nno")
         draw_singledata(ax, data, mapping, stim_map, channel_subset, dupl_electr, xindex)
    
    
    plt.show()

if __name__ == "__main__":
    main()
