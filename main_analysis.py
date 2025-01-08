import dash
import dash_bootstrap_components as dbc

import analysis_core
from CustomLogger import CustomLogger as Logger

from dashsrc.components.layout import create_layout

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipelines'))

def main():
    analysis_core.init_analysis("DEBUG")

    global_data = {"UnityTrackwise": None, "UnityFramewise": None, 'UnityTrialwiseMetrics': None,
                   "SessionMetadata":None}
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(app, global_data)
    app.run_server(host="0.0.0.0", port=8050, debug=False)

if __name__ == '__main__':
    main()