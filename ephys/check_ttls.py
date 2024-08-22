import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

PATH = '/Volumes/large/Simon/mea1k/ttl_data/recordings/'
fname = "Trace_20240618_17_02_41_withBehaviorTTL.raw.h5"

fullfname = os.path.join(PATH, fname)

with h5py.File(fullfname, 'r') as file:
    print(file.keys())
    print(file['bits'])
    print(file['bits'][:10])
    msg = np.array([(a[0],a[1]) for a in file['bits']])
    print(msg)
    plt.plot(msg[:, 0], msg[:, 1])
    plt.show()
    print(msg.shape)
    exit()
    ephys_chnls = np.array([entry[0] for entry in (file['mapping'])])
    print((ephys_chnls))

    stim_ttl = np.array(file['sig'][:, :])
    
    # drop the ephys_channels from the stim_ttl
    

# make a bunch of traces for each TTL timeseries
num_plots = 11
traces_per_plot = 100


for plot_num in range(num_plots):
    plt.figure(figsize=(18, 10))
    start = plot_num * traces_per_plot
    end = start + traces_per_plot
    for i in range(start, min(end, stim_ttl.shape[0])):
        plt.plot(stim_ttl[i, :]+100*(i-start))
        plt.text(0, stim_ttl[i, 0]+100*(i-start), f'{i}', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Plot {plot_num+1}')
    plt.show()