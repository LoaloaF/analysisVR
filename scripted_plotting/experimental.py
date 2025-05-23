import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))

import numpy as np
# from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_sessions
# from dashsrc.plot_components.plots import plot_AnimalKinematics
# from dashsrc.plot_components.plots import plot_StayRatioOverTime
# from dashsrc.plot_components.plots import plot_2DStaytimes
# from dashsrc.plot_components.plots import plot_LickTrack
from dashsrc.plot_components.plots import plot_SVMPredictions
from dashsrc.plot_components.plots import plot_SessionWaveforms
from dashsrc.plot_components.plots import plot_RawSpikes
from dashsrc.plot_components.plots import plot_rawEvents
from dashsrc.plot_components.plots import plot_fr
from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args

output_dir = "./outputs/experimental/"
data = {}
nas_dir = C.device_paths()[0]
Logger().init_logger(None, None, logging_level="DEBUG")



# # PANEL 1
# paradigm_ids = [800]
# animal_ids = [5]
# session_ids = None
# width = 715
# height = 1070
# analytic = "UnityTrackwise"
# group_by = None
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)
# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
#                                                   paradigm_ids=paradigm_ids, 
#                                                   animal_ids=animal_ids, 
#                                                   session_ids=session_ids)
# data[analytic], group_by_values = group_filter_data(data[analytic], cue_filter=["Late R"])
# fig = plot_LickTrack.render_plot(data[analytic], data['SessionMetadata'], 
#                                  width, height)
# # fullfname = f'{output_dir}/rYL001_first3Sess_kinemetics_heatmap.svg'
# # fig.write_image(fullfname, width=width, height=height, scale=1)
# # print(f"inkscape {fullfname}")
# fig.show()


# # PANEL 2
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = None
# width = 715
# height = 1070
# analytic = "UnityTrialwiseMetrics"
# group_by = None
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)
# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
#                                                   paradigm_ids=paradigm_ids, 
#                                                   animal_ids=animal_ids, 
#                                                   session_ids=session_ids)
# data[analytic], group_by_values = group_filter_data(data[analytic])
# fig = plot_StayRatioOverTime.render_plot(data[analytic], data['SessionMetadata'], 
#                                    width, height)
# # fullfname = f'{output_dir}/rYL001_first3Sess_kinemetics_heatmap.svg'
# # fig.write_image(fullfname, width=width, height=height, scale=1)
# # print(f"inkscape {fullfname}")
# fig.show()

# PANEL 3
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = [1]
# width = 715
# height = 1070
# analytic = "UnityTrialwiseMetrics"
# group_by = None
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)
# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
#                                                   paradigm_ids=paradigm_ids, 
#                                                   animal_ids=animal_ids, 
#                                                   session_ids=session_ids)
# data[analytic], group_by_values = group_filter_data(data[analytic])
# fig = plot_2DStaytimes.render_plot(data[analytic], data['SessionMetadata'], 
#                                          width, height)
# # fullfname = f'{output_dir}/rYL001_first3Sess_kinemetics_heatmap.svg'
# # fig.write_image(fullfname, width=width, height=height, scale=1)
# # print(f"inkscape {fullfname}")
# fig.show()


# # ephys
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = [25]
# # animal_ids = [10]
# # session_ids = [7,8]
# width = 1400
# height = 1400
# group_by = None
# data["Spikes"] = analytics.get_analytics('spikes', mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)
# data['SpikeClusterMetadata'] = analytics.get_analytics('SpikeClusterMetadata', mode='set',
#                                                     paradigm_ids=paradigm_ids,
#                                                     animal_ids=animal_ids,
#                                                     session_ids=session_ids)

# # session_dir = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)[0][0]
# # raw_data_mmap, mapping = session_modality_from_nas(session_dir, 'ephys_traces')
# # data['ephys_traces'] = raw_data_mmap
# # data['implant_mapping'] = mapping
# fig = plot_SessionWaveforms.render_plot(data['Spikes'], data['SpikeClusterMetadata'],
#                                         width, height,)
# fig.show()











# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# # plt.plot()
# # for i,row in enumerate(raw_data):
# #     plt.plot(i +row* -.001)
# # plt.show()

# # make a discrete colomap from 0 to 432
# cmap = plt.get_cmap('tab20', 432)

# # data["Spikes"].loc[:, 'spike_time'] = data["Spikes"].loc[:, 'spike_time'] / 50

# plt.figure(figsize=(18, 8))
# sites = []
# for cluster_id in data["Spikes"].cluster_id.unique():
#     cluster_spikes = data['Spikes'][(data["Spikes"].cluster_id == cluster_id) & (data['Spikes'].spike_time<100_000)]
#     # print(cluster_spikes)
#     if cluster_spikes.shape[0] == 0:
#         continue
#     cluster_site = cluster_spikes.spike_site.unique()[0]
#     color = cmap(cluster_site)
#     # print(cluster_site)
#     plt.scatter(cluster_spikes.spike_time,
#                 cluster_spikes.spike_site, color=color, marker='|', s=100)
#     # add vertical lines
#     # for spt in cluster_spikes.spike_time:
#     #     plt.axvline(x=spt, color=color, linestyle='--', alpha=0.3)
    
#     print()
#     for idx, row in cluster_spikes.iterrows():
#         print(idx, row.spike_time, row.spike_site)
#         print(row.spike_site + raw_data[row.spike_site, row.spike_time-10:row.spike_time+10].astype(float))
#         plt.plot(np.arange(row.spike_time-10*50,row.spike_time+10*50, 50),
#                  row.spike_site + raw_data[row.spike_site, row.spike_time//50 -10: row.spike_time//50 +10].astype(float)* -.04, color='k', alpha=0.6, linewidth=2)
#         # plt.plot(cluster_spikes.spike_time,
#         #         cluster_spikes.spike_site + raw_data[cluster_spikes.spike_site.values, cluster_spikes.spike_time].astype(float)* -.04, color=color, marker='o', facecolor='none', s=300)
        
        
    
#     # print(cluster_id)
#     sites.extend(list(cluster_spikes.spike_site.unique()))
#     # if cluster_site < raw_data.shape[0]:
#     # args = {"color":'green'} if cluster_site != 101 else {'color': 'red'}

# sites = list(set(sites))
# # sites = data["Spikes"].spike_site.unique()
# for site in sites:
#     # print(site)
    
#     col = cmap(site)
#     plt.plot(np.arange(raw_data.shape[1]) *50, 
#             site +raw_data[site].astype(float)* -.04, 
#                 linewidth=0.5, alpha=0.6,
#                 color=col)
#     # break

        
# plt.title("Cluster spikes")
# plt.ylabel("Electrode site")
# plt.xlim(0, 100_000)
# plt.ylim(160, 0)
# plt.show()

# # do a event plot of the spikes with matplotlib

# # plt.eventplot



# # ephys
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = [1]
# width = 715
# height = 1070
# group_by = None
# data["Spikes"] = analytics.get_analytics('Spikes', mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)


# # session_dir = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)[0][0]
# # raw_data_mmap, mapping = session_modality_from_nas(session_dir, 'ephys_traces')
# # data['ephys_traces'] = raw_data_mmap
# # data['implant_mapping'] = mapping
# fig = plot_rawEvents.render_plot(data['Spikes'], data['BehaviorEvents'])
# fig.show()


# # ephys
# paradigm_ids = [1100]
# animal_ids = [6]
# width = 715
# height = 1070
# # session_ids = [2]

# for s_id in range(0,33):
#     session_ids = [s_id]

#     # data["BehaviorEvents"] = analytics.get_analytics('BehaviorEvents', mode='set',
#     #                                         paradigm_ids=paradigm_ids,
#     #                                         animal_ids=animal_ids,
#     #                                         session_ids=session_ids)
#     data["FiringRateTrackbinsZ"] = analytics.get_analytics('FiringRateTrackbinsZ', mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=session_ids)
#     if data["FiringRateTrackbinsZ"] is None:
#         print(f"FiringRateTrackbinsZ not found for session {session_ids[0]}")
#         continue
#     data["UnityTrackwise"] = analytics.get_analytics('UnityTrackwise', mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=session_ids)
#     fig = plot_fr.render_plot(data['FiringRateTrackbinsZ'], data['UnityTrackwise'], 
#                               s_id=session_ids[0],)
#     fig.show()






# ephys
paradigm_ids = [1100]
animal_ids = [6]
width = 715
height = 1070
session_ids = None
session_names = ["2024-12-09_17-45_rYL006_P1100_LinearTrackStop_26min"]

svm_output = analytics.get_analytics('SVMCueOutcomeChoicePred', mode='set',
                             paradigm_ids=paradigm_ids,
                             animal_ids=animal_ids,
                             session_names=session_names,
                             session_ids=session_ids)
metad = analytics.get_analytics('SessionMetadata', mode='set',
                             paradigm_ids=paradigm_ids,
                             animal_ids=animal_ids,
                             session_names=session_names,
                             session_ids=session_ids)


fig = plot_SVMPredictions.render_plot(svm_output, metad)
fig.show()


