import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_sessions
from dashsrc.plot_components.plots import plot_AnimalKinematics
from dashsrc.plot_components.plots import plot_StayRatioOverTime
from dashsrc.plot_components.plots import plot_2DStaytimes
from dashsrc.plot_components.plots import plot_LickTrack
from analytics_processing import analytics
from CustomLogger import CustomLogger as Logger

from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data

output_dir = "./outputs/experimental/"
data = {}
Logger().init_logger(None, None, logging_level="DEBUG")



# PANEL 1
paradigm_ids = [800]
animal_ids = [5]
session_ids = None
width = 715
height = 1070
analytic = "UnityTrackwise"
group_by = None
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=paradigm_ids,
                                         animal_ids=animal_ids,
                                         session_ids=session_ids)
data['SessionMetadata'] = analytics.get_analytics('SessionMetadata', mode='set', 
                                                  paradigm_ids=paradigm_ids, 
                                                  animal_ids=animal_ids, 
                                                  session_ids=session_ids)
data[analytic], group_by_values = group_filter_data(data[analytic], cue_filter=["Late R"])
fig = plot_LickTrack.render_plot(data[analytic], data['SessionMetadata'], 
                                 width, height)
# fullfname = f'{output_dir}/rYL001_first3Sess_kinemetics_heatmap.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# print(f"inkscape {fullfname}")
fig.show()


# def update_plot(selected_paradigm, selected_animal, session_range,
#                 metric_max, outcome_filter, cue_filter, trial_filter,
#                 width, height):

#     if not all((selected_paradigm, selected_animal, metric_max)):
#         return {}
    
#     paradigm_slice = slice(selected_paradigm, selected_paradigm)
#     animal_slice = slice(selected_animal, selected_animal)
#     session_slice = [sid for sid in np.arange(session_range[0], session_range[1] + 1)
#                         if sid in global_data[analytic].index.unique('session_id')]
    
#     # paradigm, animal and session filtering
#     data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
#                                                     session_slice, :]]
#     # filter the data based on the group by values
#     data, _ = group_filter_data(data, outcome_filter, cue_filter, 
#                                 trial_filter)
    
#     sorted_by = 'session_id'
#     # sorted_by = 'cues'
#     print(data)
#     fig = plot_StayRatio.render_plot(data, metric_max, sort_by=sorted_by)
#     return fig