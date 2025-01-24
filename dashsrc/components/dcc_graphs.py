from dash import dcc

def get_general_graph_component(vis_name, fixed_height=None, fixed_width=None):
    graph_id = f'figure-{vis_name}'
    style = {}
    if fixed_height:
        style["height"] = fixed_height
    if fixed_width:
        style["width"] = fixed_width
    graph = dcc.Graph(id=graph_id,
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': f'custom_svg_image_{vis_name}',
                            # 'height': 600,
                            # 'width': 800,
                            'scale': 1
                        },
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['toImage'],
                        'modeBarButtonsToRemove': ['autoScale2d', 'zoomIn2d', 'zoomOut2d']
                    },
                    style=style)
    return graph, graph_id