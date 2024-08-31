import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


MAX_AMPL_mV = 2900.
BITRATE = 2**10
GAIN = 7
SR = 20_000
SR = 20_000
PULSEINTERVAL = 100
PULSELENGTH = 8
NPULSES = 30

# stim_traces = './impedance/rec3/output_stimulation_traces.csv'
# PATH = '/Volumes/backup/data/rec3/'
basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance"
PATH = basepath + '/device_4983/impedance_rec2/results'
    
fname = PATH + '/output_stimulation_traces.csv'
# df = pd.read_csv(stimulation_traces, index_col=(0,1,2,3), header=[0,1])
stim_traces = pd.read_pickle(
    fname.replace(".csv", ".pkl"))
print(stim_traces)


stim_set = 1
chunk = 0
elset = 0
pulse = 8

def exponential(x, a, b, c):
    return a - np.exp(c-(b*x))

# stim_traces.loc[(0,	0,	0,	slice(None)), 1].plot
for pulse in range(30):
    dat = stim_traces.loc[(stim_set, chunk, elset, slice(None)), pulse]
    dat = dat.iloc[:10]
    electrodes = [f"El{e}" for e in dat.index.get_level_values(3).values]
    plt.plot(dat.values.T, label=electrodes)
    for i in range(dat.shape[0]):
        plt.scatter((2,6), dat.iloc[i,[2,6]])
        x = np.arange(2,7)
        y = dat.iloc[i,slice(2,7)].values

        try:
            params, params_covariance = curve_fit(exponential, x, y, p0=(1414, 0.02, 3))
            xinput_fit = np.arange(2,6.1,.1)
            yfit = exponential(xinput_fit, *params)
            plt.plot(xinput_fit, yfit, color='k', alpha=.8, linestyle='--')
        except Exception as e:
            print("FAILUREEEE: ", e)
        plt.scatter(x, y, alpha=0.8, s=20, marker='o')
        
    ax = plt.gca()
    ax.hlines(MAX_AMPL_mV/2, 0, dat.shape[1], colors='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Voltage (mV) 1450 +- 207.14mV')
    ax.set_title('Current stimulation traces')
    ax.set_ylim(MAX_AMPL_mV/2 +210, MAX_AMPL_mV/2 -210)
    
    
    # output the level 3 values of the index in the order they appear
    # make a legend with the electrodes
    plt.legend(ncol= 5, loc='lower right', fontsize=6)
    
    
    plt.show()