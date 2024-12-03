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

    global_data = {"unity_trackwise": None, "unity_framewise": None}
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(app, global_data)
    app.run_server(debug=True)

if __name__ == '__main__':
    main()