import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from dashsrc.plot_components.plots import plot_SessionKinematics
from dashsrc.plot_components import kinematics_plot as kinematics_plot
from analytics_processing import analytics


analytic = 'UnityTrackwise'
data = {}
data[analytic] = analytics.get_analytics(analytic, mode='set', 
                                         animal_ids=[1,], session_ids=[0,1], paradigm_ids=[800])

n_trials = 60
trial_color = 'Outcome'
metric = 'Velocity'
metric_max = 80

# Generate the plot
data1 = data[analytic].loc[pd.IndexSlice[:,0:1,0:1]]
fig = trial_wise_kinematics_plot.render_plot(data1, n_trials, trial_color, metric, metric_max)

data2 = data[analytic].loc[pd.IndexSlice[:,0:1,1:2]]
fig2 = trial_wise_kinematics_plot.render_plot(data2, n_trials, trial_color, metric, metric_max)

# Save the plot as an image
# fig.write_image("./plot.svg", width=800, height=400)

# Display the plot
fig.show()
fig2.show()