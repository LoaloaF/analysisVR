import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

def get_delta_V(PATH):
    cached_fullfname = os.path.join(PATH, "delta_Vs.pkl")
    if os.path.exists(cached_fullfname):
        return pd.read_pickle(cached_fullfname)
    
    full_fname = PATH + '/output_stimulation_traces.pkl'
    stim_traces = pd.read_pickle(full_fname)
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
    delta_Vs.to_pickle(cached_fullfname)
    return delta_Vs

def delta_V2kOhm(delta_Vs):
    #/((7*25)/1e3) # 7nA*25 amplification  -> to mV
    # R = U/I
    # delta_Vs in mV
    # I_step = 7nA (.set_small_current_range(True)), large would be 70nA
    # amplific = 25
    # convert from mV to V, and from nA to A
    resistance = delta_Vs["mean"]/1e3 /(7*25/1e9) /1e3
    return resistance

def plot_distr(data, PATH, unit='mV', plot_std=False, device_name='', show=True):
    col = 'r'
    if unit == 'mV':
        xlabel = 'deltaV [mV]'
        title = "Mean voltage jump post stimulation"
        fname = "deltaV_mV"
        if plot_std:
            title = "STD in voltage jump post stimulation, 30 pulses"
            col = 'b'
            fname = "STD_deltaV_mV"

    elif unit == 'kΩ':
        xlabel = 'Resistance [kΩ]'
        title = "Resistive component of impedance"
        fname = "resistance_kOhm"
    else:
        raise ValueError("unit must be 'mV' or 'kΩ'")
    title += f"\n{device_name}"
    
    perc90 = np.percentile(data, 90)
    perc50 = np.percentile(data, 50)
    perc75 = np.percentile(data, 75)
    plt.hist(data, bins=100, cumulative=True, histtype='step', color=col)

    plt.vlines(perc50, 0, 26400, colors='k', linestyles='--')
    plt.text(perc50, 26400, f"50%: {perc50:.2f}{unit}", fontsize=8)
    
    plt.vlines(perc75, 0, 23400, colors='k', linestyles='--')
    plt.text(perc75, 23400, f"75%: {perc75:.2f}{unit}", fontsize=8)
    
    plt.vlines(perc90, 0, 20400, colors='k', linestyles='--')
    plt.text(perc90, 20400, f"90%: {perc90:.2f}{unit}", fontsize=8)
    
    plt.ylabel('n electrodes')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xscale('log')
    plt.savefig(f'{PATH}/{device_name.replace('/','-')}_{fname}.png')
    
    
    if show:
        plt.show()
    
def plot_heatmap(data, PATH, unit='mV', device_name='', show=True):
    if unit == 'mV':
        xlabel = 'deltaV [mV]'
        title = "Mean voltage jump post stimulation"
        vmin = 0
        vmax = 60
        fname = "deltaV_map"
    elif unit == 'kΩ':
        xlabel = 'Resistance [kΩ]'
        title = "Resistive component of impedance"
        vmin = 0
        vmax = 120
        fname = "resistance_map"
    else:
        raise ValueError("unit must be 'mV' or 'kΩ'")
    title += f"\n{device_name}"
    
    missing_els = [el_i for el_i in range(26400) if el_i not in data.index.values]
    missing_els = pd.Series(np.nan, index = missing_els)
    data = pd.concat([data, missing_els]).sort_index()

    griddata = data.values.reshape(120,220)

    # Create a colormap that sets NaN values to be transparent
    cmap = plt.cm.viridis_r  # or any other colormap
    cmap.set_bad(color='white')

    plt.figure(figsize=(18,8))
    plt.imshow(griddata, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(xlabel)
    plt.savefig(f'{PATH}/{device_name.replace('/','-')}_{fname}.png')
    if show:
        plt.show()

def main():
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_names = [
                    # 'device_4983/impedance_rec2',
                    # 'device_4984/impedance_rec2',
                    
                    # 'device_headmount_old1CornerMarked/impedance_rec2',
                    # 'device_headmount_implanted/impdance_rec2',

                    # 'device_headmount_new3EpoxyWalls/impedance_rec2_noGP',
                    'device_headmount_new2EpoxyWalls/impedance_rec2_noGP',
                    'device_headmount_new2EpoxyWalls/impedance_rec2_noGP_PBS'
                    ]
    show = True
    
    for device_name in device_names:
        PATH = os.path.join(basepath, device_name, "results")
        if not os.path.exists(PATH):
            print("Path does not exist: ", PATH)
            continue
        delta_Vs = get_delta_V(PATH)
        resitance = delta_V2kOhm(delta_Vs)
        
        # plot_distr(delta_Vs["mean"], PATH, unit="mV", device_name=device_name, show=show)
        plot_distr(resitance, PATH, device_name=device_name, unit="kΩ")
        # plot_distr(delta_Vs["std"], PATH, unit="mV", device_name=device_name, plot_std=True)
        # plot_heatmap(resitance, PATH, unit="kΩ", device_name=device_name, show=show)


if __name__ == "__main__":
    main()