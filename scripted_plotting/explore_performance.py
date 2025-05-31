import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from dashsrc.plot_components import staytimes_plot as staytimes_plot
from dashsrc.plot_components import kinematics_plot as kinematics_plot
from analytics_processing import analytics

output_dir = "./outputs/yl/"

# analytic = 'BehaviorTrialwise'
# data = {}
# # rat1_sessions = [*range(8), *range(12,20)]
# rat6_sessions = None
# data[analytic] = analytics.get_analytics(analytic, mode='set', 
#                                          animal_ids=[6], session_ids=rat6_sessions,
#                                          paradigm_ids=[1100])


# fig = staytimes_plot.render_plot(data[analytic], 10.0)
# fig.write_image(f"{output_dir}/rYL001_successrate_staytimes.pdf", height=300, width=400)
# # fig.write_image(f"{output_dir}/rYL001_successrate_staytimes.png", )
# fig.show()



analytic = 'BehaviorTrackwise'
data = {}
data[analytic] = analytics.get_analytics(analytic, mode='compute', 
                                         animal_ids=[8,], session_ids=None,
                                         paradigm_ids=[800])
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='compute', 
                                                  animal_ids=[8, ], session_ids=None,
                                                  paradigm_ids=[800])

# width = 250
# height = 190
# fig = kinematics_plot.render_plot(data[analytic], data['SessionMetadata'], 'Velocity', 80, True,
#                                   width, height)
# dpi = 300
# fig.write_image(f"{output_dir}/rYL001_kinemetics_heatmap.svg", width=width, height=height, scale=1)
# fig.write_image(f"{output_dir}/rYL001_kinemetics_heatmap.png", width=width, height=height, scale=1)
# fig.show()




# rat2_sessions = [1,2,3,4,5,9,10]
# data[analytic] = analytics.get_analytics(analytic, mode='set', 
#                                          animal_ids=[2], session_ids=rat2_sessions,
#                                          paradigm_ids=[800])

# print(data[analytic])
# n_trials = 60
# trial_color = 'Outcome'
# metric = 'Velocity'
# metric_max = 80

# Generate the plot
# data1 = data[analytic].loc[pd.IndexSlice[:,0:1,0:1]]
# fig = staytimes_plot.render_plot(data[analytic], 4.0)

# show

