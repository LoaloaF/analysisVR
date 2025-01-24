import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_sessions
from dashsrc.plot_components import kinematics_plot as kinematics_plot
from dashsrc.plot_components.plots import plot_SessionKinematics
from analytics_processing import analytics

output_dir = "./outputs/yl/"
data = {}



# PANEL 1+2, run this twice once with _draw_staytime_average one with _draw_staytime_distr
paradigm_ids = [800]
animal_ids = [8]
session_ids = None
width = 515
height = 300
analytic = "UnityTrialwiseMetrics"
max_metric = 4.0
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
fig = staytimes_plot_sessions.render_plot(data[analytic], max_metric)
fullfname = f'{output_dir}/rYL008_successrate_staytimes.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()


# PANEL 3
paradigm_ids = [800]
animal_ids = [8]
session_ids = None
# session_ids = [i for i in range(37) if i not in (5,19)]
width = 380
height = 500
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
fullfname = f'{output_dir}rYL008_first3Sess_kinemetics_heatmap.svg'
fig.write_image(fullfname, width=width, height=height, scale=1)
print(f"inkscape {fullfname}")
fig.show()


# PANEL 4
for s_id in [0, 7, 11, 27]:
    paradigm_ids = [800]
    animal_ids = [8]
    session_ids = [s_id]
    # session_ids = [i for i in range(37) if i not in (5,19)]
    width = 440
    height = 200
    analytic = "UnityTrackwise"
    max_metric = 120
    data[analytic] = analytics.get_analytics(analytic, mode='set',
                                            paradigm_ids=paradigm_ids,
                                            animal_ids=animal_ids,
                                            session_ids=session_ids)
    data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                    paradigm_ids=paradigm_ids, 
                                                    animal_ids=animal_ids, 
                                                    session_ids=session_ids)
    n_trials = data[analytic]['trial_id'].nunique()
    fig = trial_wise_kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 
                                                n_trials=n_trials, group_by='Cue', 
                                                group_by_values={'Early R': [1], 'Late R': [2]},
                                                metric='Velocity', 
                                                var_viz='80th percent.',
                                                metric_max=max_metric, 
                                                smooth_data=True,
                                                width=width, 
                                                height=height)
    fullfname = f'{output_dir}rYL008_S{s_id:02d}_kinemetics.svg'
    # fig.write_image(fullfname, width=width, height=height, scale=1)
    print(f"inkscape {fullfname}")
    fig.show()


