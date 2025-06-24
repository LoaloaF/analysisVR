import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))

import numpy as np
from dashsrc.plot_components.plots import plot_CueCorrelation

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args

output_dir = "./outputs/experimental/"
data = {}
nas_dir = C.device_paths()[0]
Logger().init_logger(None, None, logging_level="INFO")



# ephys
paradigm_ids = [1100]
animal_ids = [6]
width = 850
height = 700
session_ids = [1]

data = analytics.get_analytics('PVCueCorr', mode='set',
                                paradigm_ids=paradigm_ids,
                                animal_ids=animal_ids,
                                session_ids=session_ids)
data = data[data.predictor == 'mPFC']
metadata = analytics.get_analytics('SessionMetadata', mode='set',
                                    paradigm_ids=paradigm_ids,
                                    animal_ids=animal_ids,
                                    session_ids=session_ids)

fig = plot_CueCorrelation.render_plot(data, metadata, width=width, height=height,)
# show
fig.show()