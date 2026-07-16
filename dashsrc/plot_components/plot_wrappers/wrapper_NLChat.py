"""Natural-language module for the VR GUI (the "Ask" tab).

Same shape as any other vis module: render() returns an
html.Div(id=f"{vis_name}-container") with a control panel and an output area, and
registers a Dash callback. The difference: the control panel is a text box, and
the callback runs the LLM agent loop (dashsrc/nlchat/agent.py) whose tools reuse
the same plotting pipeline every other module uses.

Data is loaded lazily -- the tools pull from the already-loaded `global_data` if
present, else from the NAS via get_analytics -- so this module works whether or
not you pressed "Load Data" first.

NOTE: the callback is blocking (the LLM call takes a few seconds). It's wrapped in
dcc.Loading so the user sees a spinner. For a fully non-blocking UI, convert it to
a Dash background callback (needs `pip install diskcache` + a DiskcacheManager on
the app in main_analysis.py) -- see the note at the bottom of this file.
"""

from dash import html, dcc, Input, Output, State, Dash, no_update
import dash_bootstrap_components as dbc

from ...components.dcc_graphs import get_general_graph_component
from ...nlchat.agent import run_agent, MODEL, BASE_URL
import dashsrc.components.dashvis_constants as C


def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    INPUT_ID = f"nlchat-input-{vis_name}"
    SUBMIT_ID = f"nlchat-submit-{vis_name}"
    LOADING_ID = f"nlchat-loading-{vis_name}"
    RESULTS_ID = f"nlchat-results-{vis_name}"
    FIGURES_ID = f"nlchat-figures-{vis_name}"
    SUMMARY_ID = f"nlchat-summary-{vis_name}"

    figures_pane_style = {
        "paddingRight": "18px",
        "minWidth": 0,
    }

    summary_pane_style = {
        "padding": "16px 18px 18px 18px",
        "borderLeft": "1px solid rgba(80, 92, 110, 0.18)",
        "backgroundColor": "rgba(248, 250, 252, 0.78)",
        "borderRadius": "16px",
        "boxShadow": "0 1px 0 rgba(15, 23, 42, 0.03)",
        "color": "#1f2937",
        "minWidth": 0,
        "lineHeight": "1.75",
        "fontSize": "0.98rem",
    }

    loading_box_style = {
        "display": "none",
        "width": "100%",
        "margin": "0 0 14px 0",
        "alignItems": "center",
        "gap": "10px",
        "padding": "10px 14px",
        "border": "1px solid rgba(100, 116, 139, 0.16)",
        "borderRadius": "14px",
        "backgroundColor": "rgba(255, 255, 255, 0.72)",
        "color": "#475569",
    }

    results_hidden_style = {"display": "none"}

    results_visible_style = {
        "display": "block",
        "marginBottom": "18px",
    }

    @app.callback(
        Output(FIGURES_ID, "children"),
        Output(SUMMARY_ID, "children"),
        Input(SUBMIT_ID, "n_clicks"),
        State(INPUT_ID, "value"),
        prevent_initial_call=True,
        running=[
            (Output(LOADING_ID, "style"), {
                "display": "flex",
                "width": "100%",
                "margin": "0 0 14px 0",
                "alignItems": "center",
                "gap": "10px",
                "padding": "10px 14px",
                "border": "1px solid rgba(100, 116, 139, 0.16)",
                "borderRadius": "14px",
                "backgroundColor": "rgba(255, 255, 255, 0.72)",
                "color": "#475569",
            }, loading_box_style),
            (Output(RESULTS_ID, "style"), results_hidden_style, results_visible_style),
            (Output(SUBMIT_ID, "disabled"), True, False),
        ],
    )
    def on_submit(n_clicks, question):
        if not n_clicks or not question or not question.strip():
            return no_update, no_update

        figures, summary = run_agent(question.strip(), global_data)

        graphs = []
        for i, (title, fig) in enumerate(figures):
            graph, _gid = get_general_graph_component(f"{vis_name}-out{i}")
            graph.figure = fig
            graphs.append(html.Div([
                html.Small(title, style={"color": "#6b7684", "display": "block", "marginBottom": "6px"}),
                graph,
            ], style={"marginBottom": "18px"}))
        if not graphs:
            graphs = [html.Div("No figure was produced.",
                               style={"color": "#888", "padding": "20px"})]
        return graphs, summary

    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),
        html.Div([
            html.Div([
                html.H5("Ask about the data", style={"marginBottom": "12px"}),
                dcc.Textarea(
                    id=INPUT_ID,
                    placeholder="e.g. How does animal 6 change its stop/skip strategy over training on the 1D track?",
                    style={
                        "width": "100%",
                        "maxWidth": "520px",
                        "height": 92,
                        "resize": "vertical",
                        "backgroundColor": "rgba(255,255,255,0.72)",
                        "border": "1px solid rgba(100, 116, 139, 0.18)",
                        "borderRadius": "14px",
                        "padding": "12px 14px",
                    },
                ),
                dbc.Button("Ask", id=SUBMIT_ID, color="primary",
                           style={"marginTop": 12}),
                html.Small(
                    f"Model: {MODEL}  ·  endpoint: {BASE_URL}. "
                    "Set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY to change backend.",
                    style={"display": "block", "marginTop": 10, "color": "#888"},
                ),
            ], style={"padding": "8px 0 18px 0", "maxWidth": "520px", "margin": "0 0 0 0"}),
            html.Div([
                html.Div([
                    html.Span("Loading", style={"fontWeight": 600}),
                    html.Span("working on your question", style={"opacity": 0.72}),
                ], id=LOADING_ID, style=loading_box_style),
                html.Div([
                    dbc.Row([
                        # Output area: figure left, summary right, spanning the full available width.
                        dbc.Col([
                            html.Div([
                                html.Div("Figures", style={"fontSize": "0.8rem", "letterSpacing": "0.04em", "textTransform": "uppercase", "color": "#64748b", "marginBottom": "10px"}),
                                dcc.Loading(type="default", children=html.Div(id=FIGURES_ID)),
                            ], style=figures_pane_style),
                        ], width=8),
                        dbc.Col([
                            html.Div([
                                html.Div("Summary", style={"fontSize": "0.8rem", "letterSpacing": "0.04em", "textTransform": "uppercase", "color": "#64748b", "marginBottom": "10px"}),
                                dcc.Markdown(id=SUMMARY_ID, style={"margin": 0, "whiteSpace": "pre-wrap"}),
                            ], style=summary_pane_style),
                        ], width=4),
                    ], className="g-3 align-items-start"),
                ], id=RESULTS_ID, style=results_hidden_style),
            ], style={"width": "100%"}),
        ], style={"width": "100%", "margin": "0"}),
        html.Hr(),
    ], id=f"{vis_name}-container", style={"display": "none"})


# --------------------------------------------------------------------------- #
# Non-blocking upgrade (recommended once this is in real use):
#   pip install diskcache
# then in main_analysis.py:
#   from dash import DiskcacheManager
#   import diskcache
#   cache = diskcache.Cache("./.dash_cache")
#   app = dash.Dash(__name__, background_callback_manager=DiskcacheManager(cache), ...)
# and add `background=True` + a `running=[...]` (disable the Ask button while
# running) to the @app.callback above.
