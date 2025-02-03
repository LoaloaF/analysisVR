from datetime import date

import plotly.express as px
from plotly.colors import convert_colors_to_same_type

# html ids used across the app

ANIMALS = 1,2,3, 5,7, 8, 6,9
PARADIGMS = 800, 1100

SESSION_WISE_VISS = ('SessionKinematics', )
ANIMAL_WISE_VISS = ('Kinematics', 'StayPerformance', 'StayRatio', 'SessionsOverview')
DATA_LOADED_SessionKinematics_ID = 'SessionKinematics-data-loaded'
DATA_LOADED_Kinematics_ID = 'Kinematics-data-loaded'
DATA_LOADED_StayRatio_ID = 'Staytimes-data-loaded'
DATA_LOADED_StayPerformance_ID = 'Staytimes-Performance-data-loaded'
DATA_LOADED_SessionsOverview_ID = 'SessionsOverview-data-loaded'

def get_vis_name_data_loaded_id(vis_name):
    match vis_name:
        case 'SessionKinematics':
            data_loaded_id = DATA_LOADED_SessionKinematics_ID
        case 'Kinematics':
            data_loaded_id = DATA_LOADED_Kinematics_ID
        case 'StayPerformance':
            data_loaded_id = DATA_LOADED_StayPerformance_ID
        case 'StayRatio':
            data_loaded_id = DATA_LOADED_StayRatio_ID
        case 'SessionsOverview':
            data_loaded_id = DATA_LOADED_SessionsOverview_ID
        case _:
            raise ValueError(f"Unknown vis_name: {vis_name} for matching "
                            "with its data_loaded_id")
    return data_loaded_id

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

MIN_DATE_YEAR = 2024
MIN_DATE_MONTH = 7 
MIN_DATE_DAY = 1
MAX_DATE_YEAR = date.today().year
MAX_DATE_MONTH = date.today().month
MAX_DATE_DAY = date.today().day


PARADIGM_COLORS = {
    0: "#cccccc", # AutoLickReward
    100: "#666666", # Spout Association
    500: "#ff72a4", # Motor Learning
    900: "#cc72ff", # Motor Lick Learning
    1000: "#7279ff", # Motor Stop Learning
    200: "#1577d6", # 2D arena 1 Pillar
    400: "#2f15d6", # 2D arena 4 Pillars
    800: "#0ee1be", # 1D track, Cue1 Cue2
    801: "#0ee136", # 1D track, Cue1 Cue2, Lick triggered
    1100: "#d64215", # 1D track, first both rewarded, then Cue1 Cue2, Stop triggered
}

PARADIGM_VISIBLE_NAMES = {
    0: "learn-Spout-Licking",
    100: "learn-Sound->R-Association",
    500: "learn-Ball-Control",
    900: "learn-Lick-Triggers-Reward",
    1000: "learn-Complete-Stopping",
    200: "2D-Arena: One-Pillar",
    400: "2D-Arena: Four-Pillars",
    800: "1D-Track: slowing -> R1 or R2",
    801: "1D-Track: lick -> R1 or R2",
    1100: "1D-Track: stopping -> R1 and R2, then or",
}

DATA_QUALITY_COLORS = [
    [0.0, '#eb2f2f'], # -1
    [0.25, 'white'], # 0
    [0.75, '#ffdd00'], # 1
    [1.0, '#36b436'] # 2
]