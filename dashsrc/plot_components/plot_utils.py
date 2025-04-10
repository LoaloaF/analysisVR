import plotly.express as px
import plotly.colors as pc
import numpy as np

import plotly.colors as pc

def make_discr_trial_cmap(n_trials, px_seq_cmap):
    discrete_colors = pc.sample_colorscale(px_seq_cmap, samplepoints=n_trials+1)
    return dict(zip(range(n_trials+1),discrete_colors))

    # rgb = px.colors.convert_colors_to_same_type(px_seq_cmap)[0]
    # print()
    # print(px.colors.convert_colors_to_same_type(px_seq_cmap))
    # print(rgb)

    # colorscale = {}
    # # n_steps = 4  # Control the number of colors in the final colorscale
    # for i in range(len(rgb) - 1):
    #     for trial_i, step in zip(range(n_trials), np.linspace(0, 1, n_trials)):
    #     # for step in np.linspace(0, 1, n_trials):
    #         col = px.colors.find_intermediate_color(rgb[i], rgb[i + 1], step, 
    #                                                 colortype='rgb') 
    #         colorscale[trial_i] = col
    # print(colorscale)
    # return colorscale
    
def make_discr_cluster_id_cmap(cluster_ids):
    color_scale = pc.qualitative.Dark24  # You can also use D3, Set1, etc.
    colors = [color_scale[i % len(color_scale)] for i in cluster_ids]
    return {cl_id: colors[i] for i, cl_id in enumerate(cluster_ids)}