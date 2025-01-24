import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashsrc.plot_components import staytimes_plot as staytimes_plot
from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_sessions
from dashsrc.plot_components import kinematics_plot as kinematics_plot
from dashsrc.plot_components.plots import plot_SessionKinematics
from analytics_processing import analytics

def group_filter_data(data, outcome_filter, cue_filter, trial_filter, group_by="None"):
    group_values = {}
    # outcome filtering
    one_r_outcomes = [1,11,21,31,41,51,10,20,30,40,50]
    if '1 R' in outcome_filter:
        # group_values['1 R'] = [1]
        group_values['1 R'] = one_r_outcomes
    if '1+ R' in outcome_filter:
        group_values['1+ R'] = [i for i in range(1,56) if i not in one_r_outcomes]
    if 'no R' in outcome_filter:
        group_values['no R'] = [0]
    data = data[data['trial_outcome'].isin(np.concatenate(list(group_values.values())))]
    if group_by == 'Outcome':
        group_by_values = group_values
    
    # cue filtering
    group_values = {}
    if 'Early R' in cue_filter:
        group_values['Early R'] = [1]
    if 'Late R' in cue_filter:
        group_values['Late R'] = [2]
    data = data[data['cue'].isin(np.concatenate(list(group_values.values())))]
    if group_by == 'Cue':
        group_by_values = group_values
        
    # trial filtering
    group_values = {}
    # get the 1st, 2nd, 3rd proportion of trials/ split in thirds
    trial_groups = np.array_split(data['trial_id'].unique(), 3)
    if "1/3" in trial_filter:
        group_values["1/3"] = trial_groups[0]
    if "2/3" in trial_filter:
        group_values["2/3"] = trial_groups[1]
    if "3/3" in trial_filter:
        group_values["3/3"] = trial_groups[2]
    incl_trials = np.concatenate([tg for tg in group_values.values()])
    data = data[data['trial_id'].isin(incl_trials)]
    if group_by == 'Part of session':
        group_by_values = group_values
        
    if group_by == 'None':
        group_by_values = None
    
    return data, group_by_values

output_dir = "./outputs/yl/"
data = {}



# # PANEL 1
# paradigm_ids = [1100]
# animal_ids = [9]
# session_ids = [4]
# # session_ids = [i for i in range(37) if i not in (5,19)]
# width = 440
# height = 200
# analytic = "UnityTrackwise"
# max_metric = 120
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                         paradigm_ids=paradigm_ids,
#                                         animal_ids=animal_ids,
#                                         session_ids=session_ids)
# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
#                                                 paradigm_ids=paradigm_ids, 
#                                                 animal_ids=animal_ids, 
#                                                 session_ids=session_ids)
# n_trials = data[analytic]['trial_id'].nunique()
# # TODO should use filter_data functions 
# group_values = {}
# one_r_outcomes = [1,11,21,31,41,51,10,20,30,40,50]
# group_values['1 R'] = one_r_outcomes
# group_values['1+ R'] = [i for i in range(1,56) if i not in one_r_outcomes]
# group_values['no R'] = [0]
# fig = trial_wise_kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
#                                             n_trials=n_trials, group_by='Outcome', 
#                                             group_by_values=group_values,
#                                             metric='Velocity', 
#                                             var_viz='80th percent.',
#                                             metric_max=max_metric, 
#                                             smooth_data=True,
#                                             width=width, 
#                                             height=height)
# fullfname = f'{output_dir}rYL009_S{5:02d}_DR_kinemetics_outcomes.svg'
# # fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
# fig.show()






# # PANEL 2
# new_data = []
# for s_id in session_ids:
#     d, group_by_values = group_filter_data(data[analytic].loc[pd.IndexSlice[:,:,s_id:s_id]], 
#                                               trial_filter=['1/3', '3/3'], cue_filter=['Early R', 'Late R'], 
#                                               outcome_filter=['1 R', '1+ R', 'no R'], group_by='Part of session')
#     new_data.append(d)
# data[analytic] = pd.concat(new_data)
# print(group_by_values)
# print(data[analytic])
# fig = trial_wise_kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
#                                             n_trials=n_trials, group_by="Part of session", 
#                                             group_by_values=group_by_values,
#                                             metric='Velocity', 
#                                             var_viz='80th percent.',
#                                             metric_max=max_metric, 
#                                             smooth_data=True,
#                                             width=width, 
#                                             height=height)
# fullfname = f'{output_dir}rYL009_S{5:02d}_DR_kinemetics_PartOfSession.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
# fig.show()




# # PANEL 3
# paradigm_ids = [1100]
# animal_ids = [9]
# # animal_ids = [6]
# session_ids = [3,4,5,6,]
# width = 345
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)

# new_data = []
# for s_id in session_ids:
#     print(data[analytic])
#     d = group_filter_data(data[analytic].loc[pd.IndexSlice[:,:,s_id:s_id]], 
#                           trial_filter=['1/3'], cue_filter=['Early R', 'Late R'], 
#                           outcome_filter=['1 R', '1+ R', 'no R'])
#     new_data.append(d[0])
# data[analytic] = pd.concat(new_data)

# fig = staytimes_plot_sessions.render_plot(data[analytic], max_metric, var_vis='Distribution', double_reward_filter=['Early R', 'Late R'], )
# fullfname = f'{output_dir}/rYL009_DR_performance.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
# fig.show()






# # PANEL 4
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = list(range(10))
# width = 445
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8

# for reward_loc in ['Early R', 'Late R']:
#     data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=session_ids)
#     new_data = []
#     for s_id in session_ids:
#         print(data[analytic])
#         d = group_filter_data(data[analytic].loc[pd.IndexSlice[:,:,s_id:s_id]], 
#                             trial_filter=['1/3'], cue_filter=[reward_loc], 
#                             outcome_filter=['1 R', '1+ R', 'no R'])
#         new_data.append(d[0])
#     data[analytic] = pd.concat(new_data)

#     fig = staytimes_plot_sessions.render_plot(data[analytic], max_metric, 
#                                             var_vis='Distribution', 
#                                             double_reward_filter=[reward_loc, ], )
#     fullfname = f'{output_dir}/rYL006_doublerewards_{reward_loc.replace(" ","")}.svg'
#     fig.write_image(fullfname, width=width, height=height, scale=1)
#     print(f"inkscape {fullfname}")
#     fig.show()





# # PANEL 4 ++
# paradigm_ids = [1100]
# animal_ids = [9]
# session_ids = [7,10,13,28]
# width = 445
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8

# for s_id in session_ids:
#     data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=[s_id])

#     fig = staytimes_plot.render_plot(data[analytic], max_metric)
#     fullfname = f'{output_dir}/rYL009_S{s_id:02d}.svg'
#     fig.write_image(fullfname, width=width, height=height, scale=1)
#     print(f"inkscape {fullfname}")
#     fig.show()


# # PANEL 5 ++
# paradigm_ids = [1100]
# animal_ids = [9]
# session_ids = list(range(22,27))
# width = 845
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8

# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                         paradigm_ids=paradigm_ids,
#                                         animal_ids=animal_ids,
#                                         session_ids=session_ids)

# fig = staytimes_plot.render_plot(data[analytic], max_metric)
# fullfname = f'{output_dir}/rYL009_rmRIndicaterControl_S22-26.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
# fig.show()



# # PANEL 6 ++
# paradigm_ids = [1100]
# animal_ids = [6]
# session_ids = [10,11,17,18]
# width = 445
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8

# for s_id in session_ids:
#     data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=[s_id])

#     fig = staytimes_plot.render_plot(data[analytic], max_metric)
#     fullfname = f'{output_dir}/rYL006_S{s_id:02d}.svg'
#     fig.write_image(fullfname, width=width, height=height, scale=1)
#     print(f"inkscape {fullfname}")
#     fig.show()





# PANEL 7
paradigm_ids = [1100]
animal_ids = [6]
# animal_ids = [6]
session_ids = None
width = 345
height = 300
analytic = "UnityTrialwiseMetrics"
max_metric = 8
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)

new_data = []
for s_id in data[analytic].index.unique('session_id'):
    print(data[analytic])
    d = group_filter_data(data[analytic].loc[pd.IndexSlice[:,:,s_id:s_id]], 
                          trial_filter=['1/3'], cue_filter=['Early R', 'Late R'], 
                          outcome_filter=['1 R', '1+ R', 'no R'])
    new_data.append(d[0])
data[analytic] = pd.concat(new_data)

fig = staytimes_plot_sessions.render_plot(data[analytic], max_metric, var_vis='Average', double_reward_filter=['Early R', 'Late R'], )
fullfname = f'{output_dir}/rYL006_1of3EarlySess_performance.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()





















































# # PANEL 4
# paradigm_ids = [1100]
# animal_ids = [9]
# session_ids = [7,10,13,28]
# width = 445
# height = 300
# analytic = "UnityTrialwiseMetrics"
# max_metric = 8

# for s_id in session_ids:
#     data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                             paradigm_ids=paradigm_ids,
#                                             animal_ids=animal_ids,
#                                             session_ids=[s_id])

#     fig = staytimes_plot.render_plot(data[analytic], max_metric)
#     fullfname = f'{output_dir}/rYL009_S{s_id:02d}.svg'
#     fig.write_image(fullfname, width=width, height=height, scale=1)
#     print(f"inkscape {fullfname}")
#     fig.show()

# # PANEL 2
# paradigm_ids = [1100]
# animal_ids = [9]
# session_ids = None
# # session_ids = [i for i in range(37) if i not in (5,19)]
# width = 380
# height = 500
# analytic = "UnityTrackwise"
# max_metric = 1
# data[analytic] = analytics.get_analytics(analytic, mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          session_ids=session_ids)
# data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='compute', 
#                                                   paradigm_ids=paradigm_ids, 
#                                                   animal_ids=animal_ids, 
#                                                   session_ids=session_ids)
# fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
#                                   'Acceleration', max_metric, True, width, height)
# fullfname = f'{output_dir}rYL008_first3Sess_kinemetics_heatmap.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
# fig.show()


    