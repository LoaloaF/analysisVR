import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

# PATH = '/Volumes/large/Simon/mea1k/impedance/rec4_houman'
PATH = "/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_old1/impedance_rec2"
# PATH = '/Volumes/backup/data/rec3/'
fname = PATH + '/output_stimulation_traces.csv'
# fname = './impedance/rec3/output_stimulation_traces.csv'
stim_traces = pd.read_pickle(fname.replace(".csv", ".pkl"))
# print(stim_traces)

stim_traces.index = stim_traces.index.droplevel(0).droplevel(0).droplevel(0)
stim_traces = stim_traces.stack("pulse")

def deltaV(el_pulses):
    delta_v = el_pulses.iloc[:,6]- el_pulses.iloc[:,2]
    delta_v_mean = delta_v.mean()
    delta_v_std = delta_v.std()
    el = el_pulses.index[0]
    return pd.Series([delta_v_mean, delta_v_std], index=["mean", "std"], name=el)
delta_Vs = stim_traces.groupby("el").apply(deltaV)
delta_Vs.to_pickle("delta_Vs.pkl")

delta_Vs = pd.read_pickle("delta_Vs.pkl")
delta_Vs_mean = delta_Vs["mean"] #/((7*25)/1000) # 7nA*25 amplification  -> to mV


perc90 = np.percentile(delta_Vs_mean, 90)
plt.hist(delta_Vs["mean"], bins=50, )
plt.hist(delta_Vs["mean"], bins=50,  density=True, cumulative=True, histtype='step', color='r')

plt.vlines(perc90, 0, 1000, colors='k', linestyles='--')
plt.text(perc90, 1000, f"90%: {perc90:.2f}", fontsize=8)
# plt.yscale('log')
plt.ylabel('n electrodes')
plt.xlabel('delta V')
plt.title("Mean voltage jump post stimulation")
plt.xlabel('deltaV [mV]')
plt.show()

percSTD90 = np.percentile(delta_Vs["std"], 90)
plt.hist(delta_Vs["std"], bins=100, alpha=.7)
plt.vlines(percSTD90, 0, 1000, colors='k', linestyles='--')
plt.text(percSTD90, 1000, f"90%: {percSTD90:.2f}", fontsize=8)
plt.title("STD in voltage jump across 30 pulses")
plt.ylabel('n electrodes')
plt.xlabel('STD deltaV [mV]')
plt.yscale('log')
plt.show()

missing_els = [el_i for el_i in range(26400) if el_i not in delta_Vs.index.values]
missing_els = pd.Series(np.nan, index = missing_els)
delta_Vs_mean = pd.concat([delta_Vs_mean, missing_els]).sort_index()

# delta_Vs_mean = delta_Vs_mean.fillna(-50)
# delta_Vs_mean = delta_Vs_mean.fillna(delta_Vs_mean.mean())
griddata = delta_Vs_mean.values.reshape(120,220)

# Create a colormap that sets NaN values to be transparent
cmap = plt.cm.viridis_r  # or any other colormap
cmap.set_bad(color='white')

plt.figure(figsize=(18,8))
plt.imshow(griddata, cmap=cmap, vmin=0, vmax=60)
plt.title("MEA1K: Mean deltaV jump post current stimulation with Ampl=25")
cbar = plt.colorbar()
cbar.set_label('deltaV [mV]')
plt.show()


plt.figure(figsize=(18,8))
delta_Vs_mean = delta_Vs_mean.fillna(delta_Vs_mean.mean()) 
griddata = delta_Vs_mean.values.reshape(120,220)
smoothed_griddata = gaussian_filter(griddata, sigma=.7)
plt.imshow(smoothed_griddata, cmap=cmap, vmin=0, vmax=60)
plt.title("MEA1K: Mean deltaV jump post current stimulation with Ampl=25")
cbar = plt.colorbar()
cbar.set_label('deltaV [mV]')
plt.show()