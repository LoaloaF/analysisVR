import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'ephysVR'))

import dash
import dash_bootstrap_components as dbc
import argparse

from dashsrc.components.layout import create_layout
from CustomLogger import CustomLogger as Logger

def main():
    argParser = argparse.ArgumentParser("Run dash app for visualizing VR data")
    # parse the first argument as the log level
    argParser.add_argument("loglevel", help="Log level for the logger", type=str)
    args = argParser.parse_args()
    loglevel = args.loglevel
    
    Logger().init_logger(None, None, loglevel)

    loaded_analytics = {
                        # "UnityTrackwise": None, "UnityFramewise": None, 
                        "TrackKinematics": None,
                        "BehaviorTrialwise": None,
                        "BehaviorEvents": None,
                        "BehaviorFramewise": None,
                        "BehaviorTrackwise": None,
                        # 'UnityTrialwiseMetrics': None, 
                        "SessionMetadata": None, 
                        # "Portenta":None, 
                        "Spikes": None, "FiringRateTrackwiseHz": None,
                        "SpikeClusterMetadata": None, "raw_traces": None,
                        "SVMCueOutcomeChoicePred": None,
                        "PCsZoneBases": None,
                        "CCsZonewise": None,
                        "CCsZonewiseAngles": None}
    loaded_raw_traces = {}
    
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(app, loaded_analytics, loaded_raw_traces)
    # app.run(host="0.0.0.0", port=8050, debug=True)
    app.run(host="127.0.0.1", port=8000, debug=True)

if __name__ == '__main__':
    main()