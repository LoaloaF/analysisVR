import maxlab
import numpy as np
import time

# # Step 1: Set system's reference stimulation voltages
# maxlab.util.send(maxlab.system.ReferenceStimulationHigh(reference_voltage=2792))  # Set high reference to 2.25 volts
# maxlab.util.send(maxlab.system.ReferenceStimulationLow(reference_voltage=1304))   # Set low reference to 1.05 volts

# # Step 2: Set up the amplifier gain
# amplifier = maxlab.chip.Amplifier()
# amplifier.set_gain(112)  # Default gain
# maxlab.util.send(amplifier)

# # Step 3: Create a stimulation sequence
# sequence = maxlab.sequence.Sequence(initial_delay=100)

# # Step 4: Append commands for a biphasic voltage pulse
# sequence.append(maxlab.chip.DAC(0, 512-100))   # Negative pulse
# sequence.append(maxlab.system.DelaySamples(4)) # Delay of 4 samples (200 µs)
# sequence.append(maxlab.system.DelaySamples(4)) # Delay of 4 samples (200 µs)
# sequence.append(maxlab.chip.DAC(0, 512+100))   # Positive pulse
# sequence.append(maxlab.chip.DAC(0, 512))       # Return to mid-supply (0 volts)

# # Step 5: Send the sequence to the system for execution
# sequence.send()
# print("Stimulation sequence sent successfully.")

def reset_MEA1K():
	print("Resetting MEA1K...", end='', flush=True)
	maxlab.util.initialize()
	maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
	maxlab.send(maxlab.chip.Amplifier().set_gain(112))
	print("Done.")

def setup_array(electrodes, stim_electrodes=None, config_name="default_name"):
	print("Setting up array (reset,route&download)...", end='', flush=True)
	array = maxlab.chip.Array("offline")
	array.reset()
	array.clear_selected_electrodes()
	array.select_electrodes(electrodes)
	# array.connect_all_floating_amplifiers()
	# array.connect_amplifier_to_ringnode(0)

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
		# stim.set_small_current_range()
		# stim.set_large_current_range()
		stim.dac_source(0)
		maxlab.send(stim)
		time.sleep(.1)
	print("Done.")
 
def create_stim_sine_sequence(dac=0, amplitude=25, f=10, ncycles=10, nreps=3):
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

def create_stim_sequence(dac=0, amplitude=250, npulses=10, nreps=3, inter_pulse_interval=100, rep_delay_s=.1):
	def append_stimulation_pulse(seq, amplitude):
		seq.append(maxlab.chip.DAC(dac, 512-amplitude))
		seq.append(maxlab.system.DelaySamples(4))
		seq.append(maxlab.chip.DAC(dac, 512+amplitude))
		seq.append(maxlab.system.DelaySamples(4))
		seq.append(maxlab.chip.DAC(dac, 512))
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

# maxlab.send(maxlab.chip.Amplifier().set_gain(7))

# print("making it")
seq2 = create_stim_sine_sequence()
# seq3 = create_stim_sequence()
# print("making it")
# # seq2.send()

reset_MEA1K()
els = np.arange(26400).reshape(220,120)[:30,:30].flatten()
array = setup_array(els)
turn_on_stimulation_units(list(range(32)))
connect_el2stim_units(array, els[500:530])

time.sleep(5)

sequence = maxlab.sequence.Sequence(initial_delay=100)

# for i in range(100):
# 	# Step 4: Append commands for a biphasic voltage pulse
# 	sequence.append(maxlab.chip.DAC(0, 512-400))   # Negative pulse
# 	sequence.append(maxlab.system.DelaySamples(40)) # Delay of 4 samples (200 µs)
# 	sequence.append(maxlab.chip.DAC(0, 512+400))   # Positive pulse
# 	sequence.append(maxlab.chip.DAC(0, 512))       # Return to mid-supply (0 volts)
# 	sequence.append(maxlab.system.DelaySamples(40)) # Delay of 4 samples (200 µs)

# # Step 5: Send the sequence to the system for execution
# sequence.send()
seq2.send()
# print("Stimulation sequence sent successfully.")
