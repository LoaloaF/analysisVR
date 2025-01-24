import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_sessions
from dashsrc.plot_components import kinematics_plot as kinematics_plot
from analytics_processing import analytics

output_dir = "./outputs/yl/"
data = {}


# PANEL 1
paradigm_ids = [800]
animal_ids = [1]
session_ids = [1,2,3]
width = 315
height = 170
analytic = "UnityTrackwise"
max_metric = 80
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                  paradigm_ids=paradigm_ids, 
                                                  animal_ids=animal_ids, 
                                                  session_ids=session_ids)
fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
                                  'Velocity', max_metric, True, width, height)
fullfname = f'{output_dir}/rYL001_first3Sess_kinemetics_heatmap.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()



# PANEL 2
paradigm_ids = [800]
animal_ids = [1]
session_ids = list(range(4,21))
width = 380
height = 300
analytic = "UnityTrackwise"
max_metric = 1
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                  paradigm_ids=paradigm_ids, 
                                                  animal_ids=animal_ids, 
                                                  session_ids=session_ids)
fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
                                  'Acceleration', max_metric, True, width, height)
fullfname = f'{output_dir}/rYL001_laterSess_kinemetics_heatmap.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()



# PANEL 3
paradigm_ids = [800]
animal_ids = [1]
session_ids = [*list(range(8)), *list(range(11,21))] # skip 8-10, lick triggered reward sessions
width = 315
height = 300
analytic = "UnityTrialwiseMetrics"
max_metric = 4.0
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
fig = staytimes_plot_sessions.render_plot(data[analytic], max_metric)
fullfname = f'{output_dir}/rYL001_successrate_staytimes.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()


# PANEL 4
paradigm_ids = [800]
animal_ids = [2]
session_ids = list(range(3,11))
width = 380
height = 260
analytic = "UnityTrackwise"
max_metric = 80
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                  paradigm_ids=paradigm_ids, 
                                                  animal_ids=animal_ids, 
                                                  session_ids=session_ids)
fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
                                  'Velocity', max_metric, True, width, height)
fullfname = f'{output_dir}/rYL002_laterSess_kinematics.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()


# PANEL 5
paradigm_ids = [800]
animal_ids = [3]
session_ids = list(range(4,14))
width = 380
height = 260
analytic = "UnityTrackwise"
max_metric = 80
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                  paradigm_ids=paradigm_ids, 
                                                  animal_ids=animal_ids, 
                                                  session_ids=session_ids)
fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
                                  'Velocity', max_metric, True, width, height)
fullfname = f'{output_dir}/rYL003_laterSess_kinematics.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()


