"""Spikes / SpikeClusterMetadata analytics from a single JRClust run on a wireless-logger session.

Logger sessions (Intan512 logger, e.g. RUN_rPG007/rYL009_P1500/2026-09-09_15-15_...) are sorted as one JRClust run per
session, not as concatenated MEA1K batches. Session folder layout (flat):
    <session>.hdf5                          modalities (imu, orientation, camN_packages, problem_intervals, ...)
    <session>_512_ephys_traces.dat          int16, x bitScaling (0.195) = uV
    <session>_512_ephys_traces_mapping.csv  row i == .dat channel i
    <prm>.prm, <prm>_res.mat                JRClust parameters and results

Differences to extract_jrc_spikes (MEA1K):
  * JRClust sites are mapped to .dat channels via the prm siteMap (it is not the identity here).
  * spikeTimes are MATLAB 1-based -> sample_id = spikeTimes - 1 (verified on spike troughs).
  * amplitude_uV / unit_Vpp use the prm bitScaling, no /6.3 correction.
  * spikes get the zero-pad / malfunction flags from the session's problem_intervals table.
"""
import os
import re

import h5py
import numpy as np
import pandas as pd

from CustomLogger import CustomLogger as Logger

try:
    from dashsrc.plot_components.plot_utils import make_discr_cluster_id_cmap
except ImportError:
    # same as dashsrc's version (plotly qualitative.Dark24), without the dash/plotly import
    _DARK24 = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100', '#750D86',
               '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D', '#6C7C32', '#778AAE',
               '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
    def make_discr_cluster_id_cmap(cluster_ids):
        return {cl_id: _DARK24[cl_id % len(_DARK24)] for cl_id in cluster_ids}


def _parse_prm(prm_fullfname):
    txt = open(prm_fullfname, encoding='utf-8', errors='replace').read()
    def arr(key):
        m = re.search(r'^\s*' + key + r'\s*=\s*\[([^\]]*)\]', txt, re.M)
        rows = [r for r in m.group(1).split(';') if r.strip()]
        return np.array([[float(x) for x in re.split(r'[,\s]+', r.strip()) if x] for r in rows]).squeeze()
    bit_scaling = float(re.search(r'^\s*bitScaling\s*=\s*([\d.eE+-]+)', txt, re.M).group(1))
    return dict(siteMap=arr('siteMap').astype(int), shankMap=arr('shankMap').astype(int),
                bitScaling=bit_scaling)


def _interval_mask(sample_ids, intervals):
    """True where sample_id lies in any [start_sample, end_sample] (inclusive) interval."""
    a, b = intervals.start_sample.values, intervals.end_sample.values
    order = np.argsort(a); a, b = a[order], b[order]
    k = np.searchsorted(a, sample_ids, side='right') - 1
    return (k >= 0) & (sample_ids <= b[np.clip(k, 0, None)])


def extract_jrc_spikes_logger(session_fullfname, drop_cluster_types=('to_delete', '')):
    L = Logger()
    session_dir = os.path.dirname(session_fullfname)
    prm_fnames = [f for f in os.listdir(session_dir) if f.endswith('.prm') and not f.endswith('-full.prm')
                  and not f.startswith('._')]
    if len(prm_fnames) != 1:
        L.logger.warning(f"Expected 1 JRClust .prm in {session_dir}, found {prm_fnames}")
        return None, None
    prm_fullfname = os.path.join(session_dir, prm_fnames[0])
    res_fullfname = prm_fullfname[:-4] + '_res.mat'
    prm = _parse_prm(prm_fullfname)
    site2chnl = prm['siteMap'] - 1                       # JRC site (1-based) -> 0-based .dat channel, index with site-1
    site2shank = prm['shankMap']

    dat_fname = [f for f in os.listdir(session_dir) if f.endswith('ephys_traces.dat') and not f.startswith('._')][0]
    mapping = pd.read_csv(os.path.join(session_dir, dat_fname.replace('.dat', '_mapping.csv')))
    session_nsamples = os.path.getsize(os.path.join(session_dir, dat_fname)) // (2 * len(mapping))

    with h5py.File(res_fullfname, 'r') as f:
        notes = []
        for ref in f['clusterNotes'][0]:
            raw = f[ref][()]
            notes.append(''.join(map(chr, raw.ravel())) if raw.dtype == np.uint16 else '')  # empty note = uint64 [0,0]
        cluster_table = pd.DataFrame({
            "cluster_id_ssbatch": np.arange(1, len(notes) + 1),
            "cluster_type": notes,
            "unit_count": f['unitCount'][0],
            "cluster_site": f['clusterSites'][:, 0].astype(int),
            "unit_snr": f['unitSNR'][0],
            "unit_Vpp": f['unitVpp'][0],                 # uV (JRClust applies bitScaling)
            "unit_isi_ratio": f['unitISIRatio'][:, 0],
            "unit_iso_dist": f['unitIsoDist'][:, 0],
            "unit_L_ratio": f['unitLRatio'][:, 0],
        })
        spike_sites2 = f['spikeSites2'][0].astype(int)
        spikes = pd.DataFrame({
            "sample_id": f['spikeTimes'][0].astype(np.int64) - 1,          # MATLAB 1-based -> 0-based .dat row
            "cluster_id_ssbatch": f['spikeClusters'][0].astype(int),
            "jrc_site": f['spikeSites'][0].astype(int),
            "amplitude_uV": f['spikeAmps'][0] * prm['bitScaling'],         # on the ndiff-filtered trace
        })
    spikes['ephys_timestamp'] = (spikes.sample_id.values * 50).astype(np.uint64)
    spikes['channel'] = site2chnl[spikes.jrc_site.values - 1]
    spikes['channel_2nd'] = np.where(spike_sites2 > 0, site2chnl[np.clip(spike_sites2, 1, None) - 1], -1)
    spikes['channel_3rd'] = np.nan
    spikes['shank'] = site2shank[spikes.jrc_site.values - 1].astype(np.float32)
    spikes['ss_batch_id'] = 0

    cluster_table['cluster_channel'] = site2chnl[cluster_table.cluster_site.values - 1]
    cluster_table['session_nsamples'] = session_nsamples
    cluster_table['ss_batch_name'] = os.path.basename(prm_fullfname)[:-4]
    cluster_table['ss_batch_id'] = 0
    mapping_cols = ['amplifier_id', 'shank_name', 'shank_id', 'x', 'y', 'depth', 'curated_trace',
                    'gross_brain_area', 'fine_brain_area']
    cluster_table = pd.concat([cluster_table, mapping.loc[cluster_table.cluster_channel, mapping_cols]
                               .reset_index(drop=True)], axis=1)
    cluster_table = cluster_table[~np.isin(cluster_table.cluster_type, drop_cluster_types)].reset_index(drop=True)
    cluster_table['cluster_id'] = np.arange(1, len(cluster_table) + 1)
    cluster_table['cluster_id_str'] = [f"Unit{c_id:04d}" for c_id in cluster_table.cluster_id]
    cluster_table['cluster_color'] = cluster_table.cluster_id.map(make_discr_cluster_id_cmap(cluster_table.cluster_id))

    # spikes: drop hard-deleted (-1) and unassigned (0) spikes, and clusters not kept above
    spikes = spikes[spikes.cluster_id_ssbatch > 0]
    renamer = dict(zip(cluster_table.cluster_id_ssbatch, cluster_table.cluster_id))
    spikes = spikes[spikes.cluster_id_ssbatch.isin(renamer)].copy()
    spikes['cluster_id'] = spikes.cluster_id_ssbatch.map(renamer)

    # drop ISI violations (<= 1 ms to the previous spike of the same cluster), first spike is kept
    spikes = spikes.sort_values(['cluster_id', 'sample_id'], kind='stable')
    isi = np.diff(spikes.ephys_timestamp.values.astype(np.int64), prepend=-10**12)
    new_cluster = np.r_[True, spikes.cluster_id.values[1:] != spikes.cluster_id.values[:-1]]
    n_prv = len(spikes)
    spikes = spikes[new_cluster | (isi > 1_000)]

    spikes = spikes.sort_values('sample_id', kind='stable').reset_index(drop=True)
    spikes['depth'] = mapping.depth.values[spikes.channel.values]
    spikes = spikes.merge(cluster_table[['cluster_id', 'cluster_id_str', 'cluster_color', 'gross_brain_area',
                                         'fine_brain_area']], on='cluster_id', how='left')

    # flag spikes in/near logger zero pads and in the malfunction window
    try:
        problems = pd.read_hdf(session_fullfname, 'problem_intervals')
        pads = problems[problems.category.str.startswith('zero_pad')]
        sid = spikes.sample_id.values
        spikes['in_zero_pad'] = _interval_mask(sid, pads)
        near = pads.assign(start_sample=pads.start_sample - 100, end_sample=pads.end_sample + 100)
        spikes['within_5ms_of_pad'] = _interval_mask(sid, near) & ~spikes.in_zero_pad.values
        spikes['in_malfunction_window'] = _interval_mask(sid, problems[problems.category == 'logger_malfunction_window'])
    except (KeyError, FileNotFoundError, OSError):
        L.logger.warning("No problem_intervals table found, spikes are not flagged")

    cluster_table['session_spike_count'] = cluster_table.cluster_id.map(spikes.cluster_id.value_counts()).fillna(0).astype(int)
    L.logger.debug(f"Extracted {len(spikes):,} spikes from {len(cluster_table)} clusters, dropped "
                   f"{n_prv - len(spikes):,} ISI violations.\n{cluster_table}")
    return cluster_table, spikes
