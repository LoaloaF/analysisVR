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

    global_data = {"UnityTrackwise": None, "UnityFramewise": None, 'UnityTrialwiseMetrics': None,
                   "SessionMetadata": None, "Portenta":None}
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(app, global_data)
    app.run_server(host="0.0.0.0", port=8050, debug=True)

if __name__ == '__main__':
    main()