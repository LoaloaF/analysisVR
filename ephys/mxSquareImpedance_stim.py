# load the file
# voltages = sio.loadmat('2EpoxyAssemblyVoltages.mat').get('Amps')[0]
# voltages = voltages[:26400]
# import scipy.io as sio
# voltages = voltages.reshape(120,220)

# plt.imshow(voltages)
# plt.colorbar()
# plt.show()

import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import maxlab

def create_stim_sequence(dac=0, amplitude=25, npulses=10, nreps=3, inter_pulse_interval=100, rep_delay_s=.1):
    def append_stimulation_pulse(seq, amplitude):
        seq.append(maxlab.chip.DAC(0, 512-amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512+amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512))
        return seq
    
    seq = maxlab.Sequence()
    for i in range(nreps):
        for j in range(npulses):
            append_stimulation_pulse(seq, amplitude) # 25 *2.83mV - current mode?
            seq.append( maxlab.system.DelaySamples(inter_pulse_interval) ) #5ms
        time.sleep(rep_delay_s)
    return seq
    
def create_stim_sine_sequence(dac=0, amplitude=80, f=100, ncycles=10, nreps=1):
	seq = maxlab.Sequence()
  	
	# 50 us * 20kHz = 1000 samples, 1 khz exmaple
	sampling_rate = 20000  # 20 kHz
	# Create a time array
	t = np.linspace(0,1, int(sampling_rate/f))
	# Create a sine wave with a frequency of 1 kHz
	sine_wave = (amplitude * np.sin(t*2*np.pi)).astype(int) +512
	# debug = []
	for i in range(nreps):
		seq.append(maxlab.system.DelaySamples(40))
		for j in range(ncycles):
			for ampl in sine_wave:
				seq.append(maxlab.chip.DAC(dac, ampl))
				# debug.append(ampl)
	# np.save("sine_wave.npy", debug)
	# plt.show()

	return seq

def reset_MEA1K():
    print("Resetting MEA1K...", end='', flush=True)
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(7))
    print("Done.")

def setup_array(electrodes):
    print("Setting up array (reset,route&download)...", end='', flush=True)
    array = maxlab.chip.Array("offline")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    # array.connect_all_floating_amplifiers()
    # array.connect_amplifier_to_ringnode(0)
    array.route()
    array.download()
    print("Done.")
    return array

def turn_on_stimulation_units(stim_units):
    print(f"Setting up stim units {len(stim_units)}...", end="", flush=True)
    for stim_unit in stim_units:
        stim = maxlab.chip.StimulationUnit(str(stim_unit))
        stim.power_up(True)
        stim.connect(True)
        # stim.set_current_mode()
        # stim.set_small_current_range()
        # stim.set_large_current_range()
        stim.dac_source(0)
        maxlab.send(stim)
    print("Done.")
    
def start_saving(s, dir_name, fname):
    s.set_legacy_format(True)
    s.open_directory(dir_name)
    s.start_file(fname)
    s.group_delete_all()
    s.group_define(0, "all_channels", list(range(1024)))
    print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}")
    s.start_recording([0])
    
def stop_saving(s):
    print("Stopping recording...")
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()

def connect_el2stim_units(array, stim_electrodes):
    # stim_els collects electrodes that are sucessfully connected    
    stim_els, stim_units = [], []
    # failed_stim_els collects electrodes where no stimulation units could be connected to
    failed_stim_els = []
    for el in stim_electrodes:
        array.connect_electrode_to_stimulation(el)
        stim_unit = array.query_stimulation_at_electrode(el)
        
        # unknown error case, could not find routing?
        if not stim_unit:
            print(f"Warning - Could not connect El{el} to a stim unit.")
            failed_stim_els.append(el)
        
        # stim unit not used yet, 
        elif int(stim_unit) not in stim_units:
            stim_units.append(int(stim_unit))
            stim_els.append(el)
            
            if len(stim_units) == 32:
                print("Used up all 32 stim units.")
                break
        
        # stim unit already assigned case        
        else:
            array.disconnect_electrode_from_stimulation(el)
    return stim_els, stim_units, failed_stim_els
    
def new_config(i, cur_config_fromX, cur_config_fromY, sqaure_size):
    el_matrix = np.arange(26400).reshape(120,220)
    config_name = f"config_{i:02d}"
    # configuration, record from 9000 electrodes
    cur_config_toX, cur_config_toY = cur_config_fromX + sqaure_size, cur_config_fromY + sqaure_size
    el_matrix_config = el_matrix[cur_config_fromY:cur_config_toY,
                                 cur_config_fromX:cur_config_toX]
    
    reset_MEA1K()
    turn_on_stimulation_units(list(range(32)))
    return el_matrix_config, config_name

def stimulate(PATH, post_connection_sleep_time, stim_seq, sqaure_size, overlap):
    s = maxlab.Saving()
    
    cur_config_fromX = 0
    cur_config_fromY = 0
    i = 0

    electrode_mapping = pd.Series([])
    stim_mapping = pd.Series([])
    while True:
        if i >4:
            el_matrix_config, config_name = new_config(i, cur_config_fromX, 
                                                    cur_config_fromY, 
                                                    sqaure_size)
            array = setup_array(el_matrix_config.flatten())
            start_saving(s, dir_name=PATH, fname=config_name)
            
            # create a multiindex for the electrode mapping, to identify later
            midx = pd.MultiIndex.from_product([[config_name], range(el_matrix_config.size)])
            el_config_mapping = pd.Series(el_matrix_config.flatten(), index=midx)
            electrode_mapping = pd.concat([electrode_mapping, el_config_mapping])
            print(f"\n\n{config_name}... {electrode_mapping.size/26400*100:.2f}%", end=' ')


            # iterate over set of 32 stimulation electrodes
            all_failed_stim_els = []
            for stim_row_i in range(6):
                for stim_col_i in range(6):
                    stim_set_name = f"stim_set_{stim_row_i*6+stim_col_i:02d}"
                    stim_els = el_matrix_config[stim_row_i::6,stim_col_i::6]
                    print(f"{stim_set_name}: Connect {stim_els.size} electrodes to stim DAC:\n",)# stim_el)
                    
                    midx = pd.MultiIndex.from_product([[config_name], [stim_set_name], range(stim_els.size)])
                    stim_mapping = pd.concat([stim_mapping, pd.Series(stim_els.flatten(), index=midx)])
                    
                    array.select_stimulation_electrodes(stim_els.flatten())
                    stim_els, stim_units, failed_stim_els = connect_el2stim_units(array, stim_els.flatten())
                    print(f"Failed to connect these electrodes: {failed_stim_els}")
                    print(f"Downloading (actually connecting) + .1s sleep")
                    array.download() #required
                    time.sleep(post_connection_sleep_time)
                    all_failed_stim_els.extend(failed_stim_els)
                    
                    # channels = array.get_config().get_channels_for_electrodes(stim_els)
                    # channel_el_stimunit_map = np.array([channels,stim_els, stim_units]).T                
                
                    print("Stimulating ~ ~ ~ ~ ~ ~ ~ ~ ", end="")
                    stim_seq.send()
                    print("Done.")
                    
                    # time.sleep(.1)
                    print(f"Disconnecting {len(stim_els)} stimulation electrodes...", end="")
                    for stim_el in stim_els:
                        print(stim_el, end="...")
                        array.disconnect_electrode_from_stimulation(stim_el)
                    print("Done.")
                
                    # print(stim_mapping)
                    # el_matrix_config_debug = el_matrix_config.copy()
                    # el_matrix_config_debug[stim_row_i::6,stim_col_i::6] = -1
                    # plt.figure()
                    # plt.imshow(el_matrix_config_debug, vmin=el_matrix_config_debug.min()-200, vmax=el_matrix_config_debug.max())
                    # plt.show()
            print(f"Failed to connect ({len(all_failed_stim_els)}) electrodes: {all_failed_stim_els}")
            # update the current configuration, move to next square/rectangle at edges
        cur_config_fromX += sqaure_size - overlap
        if cur_config_fromX >= 220:
            cur_config_fromX = 0
            cur_config_fromY += sqaure_size - overlap
            if cur_config_fromY >= 120:
                break            
        
        stop_saving(s)
        i += 1
        # plt.imshow(el_matrix_config, vmin=0, vmax=26400)
        # if i > 3:
        # plt.show()
        
    # print(electrode_mapping)
    # print(stim_mapping)
    # print(stim_mapping.sort_values())
    return electrode_mapping, stim_mapping
    
if __name__ == "__main__":
    PATH = "/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_new2EpoxyWalls/impedance_bonded_neighbours/"
    log2file = False
    post_connection_sleep_time = .6
    if log2file:
        logfile = open(f"{PATH}/mxstimpy.log", "w")
        sys.stdout = logfile
    
    # stim_seq = create_stim_sequence()
    stim_seq = create_stim_sine_sequence()
    print("Stimulation sequence created.")
    print(stim_seq)
    
    sqaure_size = 15
    overlap = 4
    
    electrode_mapping, stim_mapping =  stimulate(PATH, post_connection_sleep_time, 
                                                 stim_seq, sqaure_size, overlap)
    # save the mappings
    electrode_mapping.to_csv(f"{PATH}/recording_mapping.csv")
    stim_mapping.to_csv(f"{PATH}/stim_mapping.csv") 