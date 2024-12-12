from dash import dcc

def get_track_graph_component(vis_name):
    graph = dcc.Graph(id=f'figure-{vis_name}',
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': f'custom_svg_image_{vis_name}',
                            'height': 600,
                            'width': 800,
                            'scale': 1
                        },
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['toImage'],
                        'modeBarButtonsToRemove': ['autoScale2d', 'zoomIn2d', 'zoomOut2d']
                    },
                    style={"height": 350})
    return graph