import dash
import dash_bootstrap_components as dbc
import pandas as pd

import analysis_core
from CustomLogger import CustomLogger as Logger

from dashsrc.components.layout import create_layout
from dashsrc.components.constants import *

analysis_core.init_analysis("DEBUG")

# Placeholder global variables for modality data
data = {m: pd.DataFrame() for m in MODALITIES}
# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout(app, data)


if __name__ == '__main__':
    app.run_server(debug=True)
