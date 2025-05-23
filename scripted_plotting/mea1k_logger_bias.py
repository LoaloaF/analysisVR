import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))

import numpy as np
import pandas as pd

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from mea1k_modules.mea1k_raw_preproc import read_raw_data

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

import matplotlib.pyplot as plt

def calc_bias(raw_data_mmap):
    chunk_size = int(.2 * 20_000) # 200 ms of data
    aggr_ADC = []
    cliped_prop = []
    for chunk_idx in range(0, raw_data_mmap.shape[1], chunk_size):
        print(chunk_idx)
        chunk_data = raw_data_mmap[:, chunk_idx:chunk_idx+chunk_size].sum(axis=1) // chunk_size
        
        cliped_prop.append(((chunk_data==0).sum(), (chunk_data==1023).sum()))
        
        hist, _ = np.histogram(chunk_data, bins=1024, range=[0,1024])
        aggr_ADC.append(hist)
        
    aggr_ADC = np.array(aggr_ADC)
    # save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "bias_logger_on_ball.npy"), aggr_ADC)
    np.save(os.path.join(output_dir, "bias_logger_on_ball_clipped_prop.npy"), np.array(cliped_prop))
    print(aggr_ADC.shape)
    print(aggr_ADC)
    
def calc_bias_stepwise(path, fname, mapping, out_fname):
    chunk_size = int(.2 * 20_000) # 200 ms of data
    downsample_after = 20_000 * 180
    skip_interval = 20_000 * 2
    aggr_ADC = []
    cliped_prop = []
    chunk_idx = -chunk_size
    i = 0
    while True:
        chunk_idx +=  chunk_size
        if chunk_idx >= downsample_after:
            chunk_idx += skip_interval
        # print(f"{i:03d}: From {chunk_idx} to {(chunk_idx+chunk_size)}")
        print(f"{i:03d}: From {chunk_idx/(20_000*60):.3f} to {(chunk_idx+chunk_size)/(20_000*60):.3f}")
        # chunk_data = read_raw_data(path, fname, convert2uV=False, 
        #                            col_slice=slice(chunk_idx, chunk_idx+chunk_size),
        #                            to_df=True)
        chunk_data = np.memmap(os.path.join(path, fname), dtype=np.int16,
                                 mode='r').reshape((738, -1), order='F')[:, chunk_idx:chunk_idx+chunk_size]
        
        if chunk_data.shape[1] != chunk_size:
            print("End of data")
            break
        chunk_data = chunk_data.sum(axis=1) //chunk_size
        cliped_prop.append(((chunk_data==0).sum(), (chunk_data==1023).sum()))
        print(cliped_prop[-1])
        
        hist, _ = np.histogram(chunk_data, bins=1024, range=[0,1024])
        aggr_ADC.append(hist)
        
        i += 1
    aggr_ADC = np.array(aggr_ADC)
    # save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f"{out_fname}.npy"), aggr_ADC)
    np.save(os.path.join(output_dir, f"{out_fname}_clipped_prop.npy"), np.array(cliped_prop))
        
    
def vis_bias(fname):
    bias = np.load(os.path.join(output_dir, fname)).T
    print(bias)
    print(bias.shape)
    
    plt.figure(figsize=(10, 4))
    img = plt.imshow(bias, aspect='auto', vmax=1)
    cbar = plt.colorbar(img)
    
    plt.title(f"Bias for {fname.replace('.npy', '')}")
    plt.xlabel("Time (s)")
    plt.xlim(906, bias.shape[1])
    # plt.xlim(0, bias.shape[1])
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [f"{int(x*0.2)}" for x in xticks])
    # plt.xlim(0, bias.shape[1])
    plt.ylim(0,bias.shape[0])
    plt.ylabel("ADC ")
    
    # get the colorbar axis
    cbar.set_ticks([0, 1])  # Example: set ticks at 0, 0.5, and 1
    cbar.set_ticklabels(['', '>1 amplifier'],)  # Example: custom labels
    plt.savefig(os.path.join(output_dir, fname.replace('.npy', '_longend.png')))
    plt.show()
    plt.close()
    
def vis_bias_clipped_prop(fname):
    bias = np.load(os.path.join(output_dir, fname))
    bias = np.concatenate((bias, (1024-bias.sum(1))[:,None]), axis=1)
    
    plt.figure(figsize=(10, 3))
    plt.stackplot(np.arange(bias.shape[0]), bias[:,0], bias[:,2], bias[:,1], 
                  labels=['< 3.2mV', 'not clipped', '> 3.2mV',],
                  colors=['red', 'green', 'orange',  ], alpha=0.7)
    plt.legend(loc='center right')
    plt.title(f"Clipping for {fname.replace('.npy', '')}")
    plt.xlabel("Time (s)")
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [f"{int(x*0.2)}" for x in xticks])
    plt.xlim(0, bias.shape[0])
    plt.ylim(0,1024)
    plt.ylabel("Clipping proportion")
    plt.savefig(os.path.join(output_dir, fname.replace('.npy', '.png')))
    plt.show()
    plt.close()
    

output_dir = "./outputs/experimental/"
data = {}
nas_dir = C.device_paths()[0]
Logger().init_logger(None, None, logging_level="INFO")




# ephys rat 10 3 minute initial logger on ball session
session_dir = sessionlist_fullfnames_from_args(paradigm_ids=[1100], animal_ids=[10], session_ids=[8])[0][0]
raw_data_mmap, _ = session_modality_from_nas(session_dir, 'ephys_traces')
calc_bias(raw_data_mmap)
vis_bias('bias_logger_on_ball.npy')
vis_bias_clipped_prop('bias_logger_on_ball_clipped_prop.npy')


# path = os.path.join(nas_dir, 'RUN_rYL010/rYL010_P1100/2025-05-01_17-28_rYL010_P1100_LinearTrackStop_42min')
# fname = 'ephys_output.raw.h5'
# mapping = pd.read_csv(os.path.join(path, "2025-05-01_17-28_rYL010_P1100_LinearTrackStop_42min_738_ephys_traces_mapping.csv"))
# # calc_bias_stepwise(path, fname, mapping, 'bias_tethered2')
# vis_bias('bias_tethered2.npy')
# vis_bias_clipped_prop('bias_tethered2_clipped_prop.npy')


# path = os.path.join(nas_dir, 'RUN_rYL010/rYL010_P0000/2025-05-09_15-51_rYL010_P0000_AutoLickReward_164min')
# fname = '2025-05-08_15-51_rYL010_P1100_FreelyMoving_3min_738_ephys_traces_mapping.dat'
# mapping = pd.read_csv(os.path.join(path, "2025-05-08_15-51_rYL010_P1100_FreelyMoving_3min_738_ephys_traces_mapping.csv"))
# calc_bias_stepwise(path, fname, mapping, out_fname='bias_logger_free')
# vis_bias('bias_logger_free.npy')
# vis_bias_clipped_prop('bias_logger_free_clipped_prop.npy')