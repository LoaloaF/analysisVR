import plotly.express as px
from plotly.colors import convert_colors_to_same_type
#   --error-color: rgb(205, 20, 20);
#   --good-color: rgb(64,202,114);
#   --outcome-0-color: var(--error-color);
#   --outcome-1-color: var(--good-color);
#   --outcome-2-color: #2cde6eff;
#   --outcome-3-color: #19ed6bff;
#   --outcome-4-color: #00ff4fff;
#   --outcome-5-color: #62ff00ff;

OUTCOME_COL_MAP = {
    0: convert_colors_to_same_type("#CD1414")[0][0],
    1: convert_colors_to_same_type("#40CA72")[0][0],
    2: convert_colors_to_same_type("#2cde6e")[0][0],
    3: convert_colors_to_same_type("#19ed6b")[0][0],
    4: convert_colors_to_same_type("#00ff4f")[0][0],
    5: convert_colors_to_same_type("#62ff00")[0][0],
}

CUE_COL_MAP = {
    0: convert_colors_to_same_type('#dbdbdb')[0][0], # unexpcted/ error
    1: convert_colors_to_same_type('#e98800')[0][0],
    2: convert_colors_to_same_type('#bc00e9')[0][0],
}

TRIAL_COL_MAP = px.colors.sequential.Viridis

MULTI_TRACES_ALPHA = .3
# MULTI_TRACES_ALPHA = int(0.1*255)  # 0 to 255