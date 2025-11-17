# !/usr/bin/env python3 
# executable - set up paths for import
import os
import sys
# to setup import paths add project root dir to sys.path (with baseVR dir in it) TODO Are the following 3 lines still relevant?
# sys.path.append(os.path.join(os.getcwd(), ".."))
# from baseVR.base_functionality import init_import_paths
# init_import_paths()

import dash
import dash_bootstrap_components as dbc
import argparse

from dashsrc.components.layout import create_layout
from CustomLogger import CustomLogger as Logger

def main():
    argParser = argparse.ArgumentParser("Run dash app for visualizing VR data")
    # parse the first argument as the log level
    argParser.add_argument("loglevel", nargs="?", help="Log level for the logger", default="INFO", type=str)
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
                        "AnalyticsOverview": None, 
                        # "Portenta":None, 
                        "Spikes": None, "FiringRateTrackwiseHz": None,
                        "SpikeClusterMetadata": None, "raw_traces": None,
                        "SVMCueOutcomeChoicePred": None,
                        "PCsZoneBases": None,
                        "CCsZonewise": None,
                        "CCsZonewiseAngles": None,
                        "PVCueCorr": None,
                        "SessionPCs40msCAs": None,
                        "SessionPCs40ms": None,
                        "Ensemble40msProjEventAligned": None,
                        }
    loaded_raw_traces = {}
    
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(app, loaded_analytics, loaded_raw_traces)
    # app.run(host="0.0.0.0", port=8050, debug=True)
    app.run(host="127.0.0.1", port=8055, debug=True)

if __name__ == '__main__':
    main()