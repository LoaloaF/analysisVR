from dash import Dash, html, dcc
from dash.dependencies import Input, Output

def render(app: Dash) -> html.Div:
    analysis_options = ['Analysis 1', 'Analysis 2', 'Analysis 3', 'Analysis 4']
    
    buttons = [
        html.Button(option, id=f"btn-{option.replace(' ', '-').lower()}", n_clicks=0)
        for option in analysis_options
    ]
    
    return html.Div(
        children=[
            html.Div(buttons, className="analysis-buttons"),
            html.Div(id='analysis-output')  # Placeholder for toggled analysis output
        ]
    )

@app.callback(
    Output('analysis-output', 'children'),
    [Input(f'btn-{option.replace(" ", "-").lower()}', 'n_clicks') for option in analysis_options]
)
def update_analysis(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Select an analysis to display."
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_analysis = button_id.split('-')[1].replace('-', ' ').title()
    
    return f"You have selected {selected_analysis}."

