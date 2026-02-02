# !/usr/bin/env python3 
# executable - set up paths for import
import os
import sys
# to setup import paths add project root dir to sys.path (with baseVR dir in it)
sys.path.append(os.path.join(os.getcwd(), "..", ))
from baseVR.base_functionality import init_import_paths
init_import_paths()

from dashsrc.plot_components.plots import plot_unit_fr_stability
from dashsrc.plot_components.plots import plot_EnsambleEncondings
from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger
from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args

from dashsrc.plot_components.plots import plot_SessionWaveforms
from dashsrc.plot_components.plots import plot_EnsembleChoiceEncoding
from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data


output_dir = "/mnt/SpatialSequenceLearning/Simon/fr_stability/"
data = {}
nas_dir = C.device_paths()[0]
Logger().init_logger(None, None, logging_level="DEBUG")

import plotly.io as pio

                                                        # 1 2 
# 3,4,5,6,7,8,9,10,11,12,13,14,
animal_ids = [6]
animal_ids = [10]
# session_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# session_ids = [5,6,7,8,9,10]
# session_ids = [11,12,13,14,]
# session_ids = [15,16,17,18,19,20,21,22,23,24,25,26]
session_ids = None
# session_ids = ['2025-04-04_18-15', '2025-04-30_16-54',]
# session_ids = ['2024-11-14_16-40', '2025-01-21_18-49']
paradigm_ids = [0, 1100]
# paradigm_ids = [1100]
# excl_session_names = ['2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min', '2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min', '2024-12-11_17-42_rYL006_P1100_LinearTrackStop_30min']
excl_session_names = None
# cols = ["session_spike_count", "session_nsamples", "cluster_id", "unit_snr", "unit_Vpp"]
cols = None
width = 700
height = 700
group_by = None

# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set',
# data['Spikes'] = analytics.get_analytics('Spikes', mode='set',
data['SpikeClusterMetadata'] = analytics.get_analytics('SpikeClusterMetadata', mode='set',
                                                       columns = cols,
                                                       paradigm_ids=paradigm_ids,
                                                       animal_ids=animal_ids,
                                                       excl_session_names=excl_session_names,
                                                       session_ids=session_ids)

data['SpikeClusterMetadata'] = data['SpikeClusterMetadata'][data['SpikeClusterMetadata']['shank_id'] > 1]
data['SpikeClusterMetadata'].sort_index(level=['session_id'], inplace=True)

# data['SpikeClusterMetadata'].index.droplevel(['entry_id', 'animal_id']).drop_duplicates()
# data['SessionMetadata']
fig = plot_SessionWaveforms.render_plot(data['SpikeClusterMetadata'], 
                                        # data['SessionMetadata'], 
                                        width=1200, height=800
                                        )
fullfname = f'{output_dir}/waveforms_rat10.html'
fig.write_html(fullfname)
print(f"Saved figure to {fullfname}")




