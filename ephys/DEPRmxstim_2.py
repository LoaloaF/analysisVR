# #!/usr/bin/python

import sys
import os
sys.path.insert(0, '/home/Alexei/nt_simon_nas/MaxLab/toolboxes/python/modules/')

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

import random
import time
import numpy as np

# from maxlab.saving import Saving

def reset_MEA1K():
	print("Resetting MEA1K...", end='', flush=True)
	maxlab.util.initialize()
	maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
	maxlab.send(maxlab.chip.Amplifier().set_gain(7))
	print("Done.")

def setup_array(electrodes, stim_electrodes=None, config_name="default_name"):
	print("Setting up array (reset,route&download)...", end='', flush=True)
	array = maxlab.chip.Array("offline")
	array.reset()
	array.clear_selected_electrodes()
	array.select_electrodes(electrodes)
	if stim_electrodes is not None:
		array.select_stimulation_electrodes(stim_electrodes)
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
		stim.set_current_mode()
		stim.dac_source(0)
		maxlab.send(stim)
		time.sleep(.1)
	print("Done.")

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

def connect_el2stim_units(array, stim_electrodes):
	# stim_els collects electrodes that are sucessfully connected	
	stim_els, stim_units = [], []
	# failed_stim_els collects electrodes where no stimulation units could be connected to
	failed_stim_els = []
	for el in stim_electrodes:

		array.connect_electrode_to_stimulation(el)

		stim_unit = array.query_stimulation_at_electrode(el)

		
		if not stim_unit:
			print(f"Warning - Could not connect El{el} to a stim unit.")
			failed_stim_els.append(el)
			#array.disconnect_electrode_from_stimulation(el)
		
		# stim unit not used yet, 
		elif int(stim_unit) not in stim_units:
			#print(f"El{el:05} -> stim unit{int(stim_unit):02}")
			stim_units.append(int(stim_unit))
			stim_els.append(el)
			
			if len(stim_units) == 32:
				print("Used up all 32 stim units.")
				break
		
		# stim unit already assigned case		
		else:
			#print(f"El{el} -> stim unit {stim_unit} already assigned.")
			array.disconnect_electrode_from_stimulation(el)
	return stim_els, stim_units, failed_stim_els

def main():
	random.seed(1)
	np.random.seed(1)
	PATH = './impedance/rec3'
	logfile = open(f"{PATH}/mxstimpy.log", "w")
	#sys.stdout = logfile

	all_els = np.arange(26400)
	nsets = len(all_els)//1024 +1

	stim_seq = create_stim_sequence()
	#reset_MEA1K()
	#turn_on_stimulation_units(list(range(32)))
	for el_set_i in range(nsets):
		t0 = time.time()
		
		# sample 1024 electrodes, remove sampled ones for unique samples
		size = 1024 if el_set_i != nsets-1 else len(all_els)%1024
		el_smple_idx = np.random.choice(np.arange(len(all_els)), 
						size=size, replace=False)
		el_smple = all_els[el_smple_idx]
		all_els = np.delete(all_els, el_smple_idx)
		
		reset_MEA1K()
		turn_on_stimulation_units(list(range(32)))

		stim_set_name = f"stim_set_{el_set_i:02}"
		stim_set_path = PATH + f'/{stim_set_name}'
		os.mkdir(stim_set_path)
		array = setup_array(el_smple, stim_electrodes=None)#, config_name=stim_set_name)
		array.save_config(f"{stim_set_path}/mxConfig.cfg")
		
		
		chunk_size = 256
		for chunk_i, stim_el_idx in enumerate(range(0, len(el_smple), chunk_size)):
			stim_el_smple_failed = []			
			stim_el_smple = el_smple[stim_el_idx:stim_el_idx+chunk_size]
			print(f"\n\nStimulation set: {el_set_i} - chunk {stim_el_idx//chunk_size} - chunksize={len(stim_el_smple)}")
			print(f"Trying to connect these electrodes to stimulation units in chunks of 32:\n{stim_el_smple}")
			array.select_stimulation_electrodes(stim_el_smple)		
			#array = setup_array(el_smple, stim_el_smple, stim_set_name)
			#array.download()
			#array.save_config(f"{stim_set_path}/mxConfig_chunk{chunk_i}.cfg")

			j = 0
			while True:
				print(f"---\nTrying to connect remaining {len(stim_el_smple)} electrodes to stim units...")
				stim_els, stim_units, failed_stim_els = connect_el2stim_units(array, stim_el_smple)
				print(f"Failed to connect these electrodes: {failed_stim_els}")
				print(f"Downloading (actually connecting) + 1s sleep")
				array.download() #required
				time.sleep(.6)

				
				# remove stim_els from stim_el_smple 
				indices= []
				for el in stim_els:
					indices.append(np.where(stim_el_smple==el)[0][0])
				# also remove the failed ones - TODO: they should be part of the new chunk
				for el in failed_stim_els:
					indices.append(np.where(stim_el_smple==el)[0][0])

				stim_el_smple = np.delete(stim_el_smple, indices)

				if not stim_units:
					print("No stim unit assigned -> chunk done.")
					break

				fname = f"{stim_set_path}/mxConfig_chunk{chunk_i}_map{j:02}.npy"
				channels = array.get_config().get_channels_for_electrodes(stim_els)
				channel_el_stimunit_map = np.array([channels,stim_els, stim_units]).T
				print(f"stim_set:{el_set_i}-chunk:{chunk_i}-elset:{j}")
				print("Channel-Electrode-StimUnit:")				
				print(channel_el_stimunit_map)				
				np.save(fname, channel_el_stimunit_map)
				# np.save(fname, np.array([stim_els, stim_units]).T)
				# fname = f"{stim_set_path}/mxConfig_chunk{chunk_i}_tstamp{j:02}.npy"
				# np.save(fname, np.array([t0, t1]))


				print("_|_|_|_|_|_|_|_")
				print("Sending pulses.")
				stim_seq.send()
				print("_|_|_|_|_|_|_|_")

				print(f"Disconnecting {len(stim_els)} stimulation electrodes...", end="")
				for stim_el in stim_els:
					print(stim_el, end="...")
					array.disconnect_electrode_from_stimulation(stim_el)
				print("Done.")				
				j += 1
				time.sleep(.5)
		
		array.close()
		print(f"Electrode set  {el_set_i} finished. Please Stop and Restart saving. Press Enter to continue")
		input()

#		if el_set_i == 1:
#			logfile.close()
#			sys.stdout = sys.__stdout__
#			return

if __name__ == "__main__":
    main()
































def stim_electrodes(electrodes, i):
	print(f"Stimulation set: {i}")

	#saver = Saving()
	#saver.set_legacy_format(True)
	#saver.start_file(f"/home/Alexei/nt_simon_nas/rec11/stimu_set_{i}")

	#stimulation_electrodes = random.sample(electrodes, 1024)
	stimulation_electrodes = electrodes
	final_stimulation_electrodes = []

	array = maxlab.chip.Array(f'stimulation_{i}')
	array.reset()
	array.clear_selected_electrodes( )
	array.select_electrodes( electrodes )
	array.select_stimulation_electrodes( stimulation_electrodes )
	print("routing...", end='')
	array.route()
	print("done.")

	array.disconnect_electrode_from_stimulation()

	######################################################################
	# 2. Connect stimulation units to the stimulation_electrodes
	######################################################################
	stimulation_units = []
	print("Connecting stimulation electrodes...", end="")
	for stim_el in stimulation_electrodes:
		array.connect_electrode_to_stimulation( stim_el )
		stim = array.query_stimulation_at_electrode( stim_el )
		if stim:
			if int(stim) not in stimulation_units:
				stimulation_units.append( int(stim) )
				final_stimulation_electrodes.append( stim_el )
				print(f"El{stim_el:05} -> StimUnit{int(stim):02}")
				if len(stimulation_units) == 32:
					print("Mapped all stimulation units.")
					break
		else:
			print(f"El{stim_el} -> Could not connect!")

	print("downloading...", end="")
	array.download()
	print("done.")
	array.save_config(f"mxConfigs/stimconfig_{i}.cfg")
	stim_config = np.array([stimulation_units, final_stimulation_electrodes])
	np.save(f'mxConfigs/stimconfig_{i}.npy', stim_config)

	#print("Starting to save")
	#saver.start()

	#print("offsetting...", end="", flush=True)
	#maxlab.util.offset()
	#print("done.", flush=True)
	#exit()


	######################################################################
	# 3. Power up and configure the stimulation units
	######################################################################

	print("Setup stim units")
	stimulation_unit_commands = []
	stimulation_units = np.sort(np.unique(stimulation_units))

	

	def append_stimulation_pulse(seq, amplitude):
		seq.append( maxlab.chip.DAC(0, 512-amplitude) )
		seq.append( maxlab.system.DelaySamples(4) )
		seq.append( maxlab.chip.DAC(0, 512+amplitude) )
		seq.append( maxlab.system.DelaySamples(4) )
		seq.append( maxlab.chip.DAC(0, 512) )
		return seq

	for stimulation_unit in stimulation_units:
		print(f"Setting up stim unit {stimulation_unit}...", end="")
		# Stimulation Unit
		stim = maxlab.chip.StimulationUnit(str(stimulation_unit))
		stim.power_up(True)
		stim.connect(True)
		stim.set_current_mode()
		#stim.set_voltage_mode()
		stim.dac_source(0)
		stimulation_unit_commands.append(stim)
		maxlab.send(stim)
		time.sleep(3)
		print("Done - ", end='')

		seq = maxlab.Sequence()
		for _ in range(3):
			for rep in range(10):
				append_stimulation_pulse(seq, 25) # 25 *2.83mV - current mode?
				seq.append( maxlab.system.DelaySamples(100) ) #5ms
			
			print("50 pulses", end="..", flush=True)
			seq.send()
			time.sleep(.2)
		print(f"Turning off stim unit {stimulation_unit}")
		stim.power_up(False)
		stim.connect(False)
		maxlab.send(stim)


	#Send 10 pulses simultaneously to the 32 electrodes
	#seq = maxlab.Sequence()
	#print("Sending pulses",end="")
	#for rep in range(1,300):
	#	append_stimulation_pulse(seq, 25)
	#	seq.append( maxlab.system.DelaySamples(200) )
	#seq.send()

	#saver.stop_file()
	#saver.stop_recording()
	#print("\n\n")

