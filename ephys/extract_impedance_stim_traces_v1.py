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
PULSEINTERVAL = 100
PULSELENGTH = 8
NPULSES = 30
    
def get_n_stimsets(path):
    return len(glob(f"{path}/stim_set_*"))

def get_n_chunks_for_stimset(stim_set_dir):
    return len(glob(f"{stim_set_dir}/mxConfig_chunk*_map00.npy"))

def get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i):
    return len(glob(f"{stim_set_dir}/mxConfig_chunk{chunk_i}_map*.npy"))

def get_resolution():
    return (MAX_AMPL_mV/BITRATE) /GAIN

def convert_to_vol(data):
    # scale data to mv
    res = get_resolution()
    data = (data-BITRATE/2) *res
    # floating/ offset
    data += MAX_AMPL_mV/2 
    return data

def read_stim_map(path, chunk_i, j):
    stimmap_fname = f'mxConfig_chunk{chunk_i}_map{j:02d}.npy'
    return np.load(os.path.join(path, stimmap_fname), 'r').astype(int)
    
# def butter_highpass_filter(data, cutoff, fs, order=5):
#     def butter_highpass(cutoff, fs, order=5):
#         nyq = 0.5 * fs
#         normal_cutoff = cutoff / nyq
#         b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
#         return b, a
#     b, a = butter_highpass(cutoff, fs, order=order)
#     y = signal.filtfilt(b, a, data)
#     return y

def extract_stim_events(path, precomputed=None, debug=True):
    n_stim_sets = get_n_stimsets(path)
    ttls = {}
    for stim_set_i in range(n_stim_sets):
        print(f"Extracting stim TTLs for stim_set_{stim_set_i:02d}")
        
        stim_set_fname = os.path.join(path, 'recordings', f"stim_set_{stim_set_i:02d}.raw.h5")
        stim_set_file = h5py.File(stim_set_fname, 'r')
        
        # chunk_i (4), elset_i (11-13), pulse_i (30)
        precomputed_fname = stim_set_fname.replace("raw.h5", "stim_ttls.pkl")
        if precomputed == "from":
            ttls = {**ttls, **pickle.load(open(precomputed_fname, 'rb'))}
            continue
        
        with h5py.File(stim_set_fname, 'r') as file:
            stim_ttl = np.array(stim_set_file['sig'][1024, :])

        diffdata = np.diff(stim_ttl)
        t = np.arange(diffdata.size)[diffdata==65511]
        # t = np.arange(diffdata.size)[diffdata==65525]

        stim_set_dir = os.path.join(path, f"stim_set_{stim_set_i:02d}")
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
            plt.title(f"Stim set {stim_set_i}")
            plt.plot(diffdata)
            for k, v in ttls.items():
                if k[0] == stim_set_i:
                    plt.text(v, stim_ttl[v], f"{k}", fontsize=8)
            plt.show()
    return ttls

def extract_eletrode_pulses(path, stim_ttls, debug=True):
    def process_elset():
        # (32,2) mapping from channel (1028) to stim channel electrode (26400)
        # channel_map = np.stack([mapping[(mapping[:,1] == stim_el)] for stim_el in stim_map[0]])[:,0]
        # channel_map = channel_map[np.argsort(channel_map[:,0])]
        index = [(stim_set_i, chunk_i, elset_i, stim_el) for stim_el in elset_stim_map[:,1]]
        index = pd.MultiIndex.from_tuples(index, names=["stim_set", "chunk", "elset", "el"])
        
        pulse_data = []
        for pulse_i in range(NPULSES):
            # get the t index of a single pulse
            tstart = stim_ttls[stim_set_i, chunk_i, elset_i, pulse_i]
            tend = tstart+(PULSELENGTH+PULSEINTERVAL)
            col_idx = pd.MultiIndex.from_product([[pulse_i],range(tend-tstart)], 
                                                 names=["pulse", "t"])
            # read a single pulse from the hdf5 file at at all the electrodes of this el set (1-32)
            pulse_dat = np.array(stim_set_file['sig'][elset_stim_map[:,0], tstart:tend])
            pulse_dat = convert_to_vol(pulse_dat)
            pulse_data.append(pd.DataFrame(pulse_dat, index=index, columns=col_idx))
        elset_data = pd.concat(pulse_data, axis=1)
        
        # save data in stim_set_dir
        # stim_set_output_dir = os.path.join(path, f"output_stim_set_{stim_set_i:02d}")
        # if not os.path.exists(stim_set_output_dir):
        #     os.mkdir(stim_set_output_dir)
        # elset_data.to_csv(os.path.join(stim_set_output_dir, f"imped_chunk{chunk_i}_elset_{elset_i:02d}.csv"))
        
        # debugging
        if debug:
            print("Channel map:\n", elset_stim_map)
            print("tstamps: ", tstart, tend)
            t_context_length = 30000
            fromt = max(0, tstart-t_context_length)
            tot = min(stim_set_file['sig'].shape[1], tstart+t_context_length)
            elset_data = np.array(stim_set_file['sig'][elset_stim_map[:,0], fromt:tot])
            # elset_data = np.array(stim_set_file['sig'][0:32, fromt:tot])
            elset_data = convert_to_vol(elset_data)
            
            t_index = np.arange(fromt, tot)
            plt.plot(t_index, elset_data.T, label=elset_stim_map[:,0], alpha=.4)
            #draw vertical lines at tstart and tend
            plt.axvline(x=tstart, color='k', alpha=.5, linestyle='--')
            plt.axvline(x=tend, color='k', alpha=.5, linestyle='--')
            plt.ylabel("Voltage (mV)")
            # horizontal line at 1450
            plt.axhline(y=MAX_AMPL_mV/2, color='k', alpha=.5, linestyle='--')
            plt.title(f"Stim set {stim_set_i}, chunk {chunk_i}, elset {elset_i} (n={elset_stim_map.shape[0]})")
            # plt.legend(ncol=4, loc='upper right')
            plt.ylim(MAX_AMPL_mV/2 +210, MAX_AMPL_mV/2 -210)
            plt.show()
        return elset_data
         
    agg_data = []
    n_stim_sets = get_n_stimsets(path)
    for stim_set_i in range(0,n_stim_sets):
        stim_set_fname = os.path.join(path, 'recordings', f"stim_set_{stim_set_i:02d}.raw.h5")
        stim_set_file = h5py.File(stim_set_fname, 'r')

        print(f"\n\n--------Processing stimulation set {stim_set_i}---------")
        stim_set_dir = f"stim_set_{stim_set_i:02d}"
        stim_set_dir = os.path.join(path, stim_set_dir)
        for chunk_i in range(get_n_chunks_for_stimset(stim_set_dir)):
            print(f"\nchunk {chunk_i}:")
            # classic 1028 channels mapping to electrode (max 26400)
            # mapping = read_mxconfig(stim_set_dir, -1)
            #mapping = read_mxconfig(os.path.join(rec_dir, "stim_set_00"), -1)
            # print(mapping[:18])

            n_elsets = get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i)
            for elset_i in range(n_elsets):
                print(f"elset{elset_i}",end="...")
                # (32,2) mapping from electrode (26400) to stim channel (32)
                elset_stim_map = read_stim_map(stim_set_dir, chunk_i, elset_i)
                agg_data.append(process_elset())
                
        stim_set_file.close()
    agg_data = pd.concat(agg_data)
    agg_data.to_csv(os.path.join(path, "output_stimulation_traces.csv"))
    agg_data.to_pickle(os.path.join(path, "output_stimulation_traces.pkl"))

def main():
    # PATH = '/run/media/loaloa/backup/data/rec3/'
    # PATH = '/Volumes/backup/data/rec3/'
    PATH = '/Volumes/large/Simon/mea1k/impedance/rec4_houman'
    stim_ttls = extract_stim_events(PATH, precomputed='from', debug=False)
    extract_eletrode_pulses(PATH, stim_ttls, debug=True)

if __name__ == "__main__":
    main()
    
# TODO IMPORTANT & DIFFICULT: find a way to set the stim unit bias to baseline 
# TODO IMPORTANT & DIFFICULT: find out what current we are setting 
# TODO save using API instead of manual
# TODO record another dataset and compare the new values
# TODO double the amplitude and check the values
# TODO investigate non-synchronized connection/diconnection to stim units 
# TODO extend electrode selection algorithm to electrodes that failed to connect 

# TODO write the UFTE <-> MEA1K alignment optimizer using dummy impedance data