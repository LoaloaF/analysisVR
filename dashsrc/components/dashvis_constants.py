import plotly.express as px
from plotly.colors import convert_colors_to_same_type

# html ids used across the app

ANIMALS = 1,2,3, 5,7, 8, 6,9
PARADIGMS = 800, 1100

SESSION_WISE_VISS = ('SessionKinematics', )
ANIMAL_WISE_VISS = ('Kinematics', 'StayPerformance', 'StayRatio')
DATA_LOADED_SessionKinematics_ID = 'SessionKinematics-data-loaded'
DATA_LOADED_Kinematics_ID = 'Kinematics-data-loaded'
DATA_LOADED_StayRatio_ID = 'Staytimes-data-loaded'
DATA_LOADED_StayPerformance_ID = 'Staytimes-Performance-data-loaded'

OUTCOME_COL_MAP = {
    0: convert_colors_to_same_type("#CD1414")[0][0],
    1: convert_colors_to_same_type("#40CA72")[0][0],
    2: convert_colors_to_same_type("#2cde6e")[0][0],
    3: convert_colors_to_same_type("#19ed6b")[0][0],
    4: convert_colors_to_same_type("#00ff4f")[0][0],
    5: convert_colors_to_same_type("#62ff00")[0][0],
}
extension = {outc: OUTCOME_COL_MAP[max(outc//10, outc%5)] for outc in range(6, 56)}
OUTCOME_COL_MAP.update(extension)

CUE_COL_MAP = {
    0: convert_colors_to_same_type('#dbdbdb')[0][0], # unexpcted/ error
    1: convert_colors_to_same_type('#e98800')[0][0],
    2: convert_colors_to_same_type('#bc00e9')[0][0],
}

FLAT_GRAY_COL_MAP = { # cue column is used as dummy column
    0: convert_colors_to_same_type('#888888')[0][0], # unexpcted/ error
    1: convert_colors_to_same_type('#888888')[0][0],
    2: convert_colors_to_same_type('#888888')[0][0],
}

REWARD_LOCATION_COL_MAP = {
    1: 'rgba(120,120,120, 1)',
    2: 'rgba(190,190,190, 1)',
}
EARLY_REWARD_LOCATION_COLOR = 'rgba(120,120,120, 1)'
LATE_REWARD_LOCATION_COLOR = 'rgba(190,190,190, 1)'
    

TRIAL_COL_MAP = px.colors.sequential.Viridis

MULTI_TRACES_ALPHA = .3
MULTI_MARKERS_ALPHA = .6
# MULTI_TRACES_ALPHA = int(0.1*255)  # 0 to 255

KINEMATICS_HEATMAP_DEFAULT_HEIGHT = 450
KINEMATICS_HEATMAP_DEFAULT_WIDTH = 600
TRACK_VISUALIZATION_HEIGHT = 30
KINEMATICS_HEATMAP_XLABELSIZE_HEIGHT = 40