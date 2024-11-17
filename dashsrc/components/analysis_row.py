from dash import dcc, html
import dash_bootstrap_components as dbc

# Helper function to create a row with controls and plots
def create_analysis_row(row_index):
    return html.Div([
        dbc.Row([
            # Left side for plots
            dbc.Col([
                dcc.Graph(
                    id=f'main-plot-{row_index}', 
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': f'custom_svg_image_{row_index}',
                            'height': 600,
                            'width': 800,
                            'scale': 1
                        },
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['toImage'],
                        'modeBarButtonsToRemove': ['autoScale2d', 'zoomIn2d', 'zoomOut2d']
                    }
                ),
                dcc.Graph(id=f'secondary-plot-{row_index}')
            ], width=9),

            # Right side for UI controls
            dbc.Col([
                html.H5(f"Data Selection for Row {row_index + 1}", style={"marginTop": 20}),
                
                # Dropdown for animal selection
                dcc.Dropdown(
                    id=f'animal-dropdown-{row_index}',
                    options=[{'label': f'Animal {i}', 'value': i} for i in [1, 2, 3, 8]],
                    multi=True,
                    placeholder="Select one or more animals"
                ),
                
                # Dropdown for session selection
                dcc.Dropdown(
                    id=f'session-dropdown-{row_index}',
                    multi=True,
                    placeholder="Select session(s)"
                ),
                
                # Dropdown for trial selection
                dcc.Dropdown(
                    id=f'trial-dropdown-{row_index}',
                    multi=True,
                    placeholder="Select trial(s)"
                )
            ], width=3)
        ]),
        html.Hr()
    ], id=f'row-{row_index}', style={'display': 'none'})  # Initial state is hidden

