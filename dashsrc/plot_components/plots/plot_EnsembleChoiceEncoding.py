import plotly.graph_objects as go
from collections import defaultdict
import pandas as pd
from plotly.subplots import make_subplots

import plotly.colors


def _get_session_ids(encoding_data):
    idx = encoding_data.index
    if isinstance(idx, pd.MultiIndex) and 'session_id' in idx.names:
        return idx.unique('session_id')
    if idx.name == 'session_id':
        return idx.unique()
    if 'session_id' in encoding_data.columns:
        return pd.Index(encoding_data['session_id'].unique())
    return idx.unique()


def _format_session_id(session_id):
    try:
        return f"{int(session_id):02d}"
    except (TypeError, ValueError):
        return str(session_id)


def activation_to_color(activation, vmin=0.6, vmax=1.4):
    # Normalize activation to [0, 1]
    norm = (activation - vmin) / (vmax - vmin)
    norm = min(max(norm, 0), 1)  # Clip to [0, 1]
    # Get color from plasma colormap
    color = plotly.colors.sample_colorscale('tempo', [norm])[0]
    return color

def compute_node_values(data):
    node_value = defaultdict(float)
    for path, val in data.items():
        for node in path:
            node_value[node] += val
    return node_value


def organize_nodes_by_column(node_meta, node_value):
    columns = defaultdict(list)
    for node_id, meta in node_meta.items():
        x = meta["x"]
        columns[x].append((node_id, node_value[node_id]))
    return columns


def compute_y_positions(columns, node_meta, pad=0.02):
    
    # Force these nodes to the top of their column
    forced_top = {
        0.4: 'R1',
        0.8: 'R2',
    }


    final_nodes = []
    for x in sorted(columns.keys()):
        group = columns[x]

        # Separate forced-top node (if specified) and others
        forced = [(nid, val) for nid, val in group if forced_top.get(x) == nid]
        others = [(nid, val) for nid, val in group if nid != forced_top.get(x)]

        # Sort the remaining nodes by descending value
        others = sorted(others, key=lambda x: -x[1])
        ordered_group = forced + others

        # Compute normalized vertical positions
        total = sum(val for _, val in ordered_group)
        y_cursor = 0.0
        for node_id, val in ordered_group:
            height = val / total * (1 - pad * (len(ordered_group) - 1))
            y_center = y_cursor + height / 2
            print(f"Node {node_id} at x={x}, y_center={y_center}, height={height}")
            
            node_col = node_meta[node_id]["color"]
            # if node_id == 'Cue1':
            #     node_col = 'orange'
            # elif node_id == 'Cue2':
            #     node_col = 'purple'
            # else:
            final_nodes.append({
                "id": node_id,
                "label": node_meta[node_id]["label"],
                "color": node_col,
                "x": node_meta[node_id]["x"],
                "y": y_center,
                "height": height
            })
            y_cursor += height + pad

    return final_nodes


def build_links(data, label_index, color_outcome, WIN_COLOR, LOOSE_COLOR):
    sources, targets, values, link_colors = [], [], [], []

    for cue, r1, r2 in data:
        val = data[(cue, r1, r2)]

        # Cue1 → R1 or notR1
        sources.append(label_index[cue])
        targets.append(label_index[r1])
        values.append(val)
        col = 'rgba(128,128,128,0.)' # light light gray
        if color_outcome: 
            if (cue == 'Cue1' and r1 == 'R1'):
                col = WIN_COLOR
            elif (cue == 'Cue1' and r1 == 'notR1'):
                col = LOOSE_COLOR
        link_colors.append(col)

        # R1 or notR1 → R2 or notR2
        sources.append(label_index[r1])
        targets.append(label_index[r2])
        values.append(val)
        col = 'rgba(128,128,128,0.)' # light light gray
        if color_outcome: 
            if (cue == 'Cue2' and r2 == 'R2'):
                col = WIN_COLOR
            elif (cue == 'Cue2' and r2 == 'notR2'):
                col = LOOSE_COLOR
        link_colors.append(col)

    return sources, targets, values, link_colors


def build_sankey_figure(nodes, sources, targets, values, link_colors, domain_y=None):
    # Set inside color to transparent, and edge color to link_colors
    print(link_colors)
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=15,
            thickness=15,
            hoverinfo='none',
            line=dict(color='black', width=0.5),
            color=[n["color"] for n in nodes],
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            line=dict(color="rgb(0,0,0)", width=0.5),
            color=link_colors,       # colored edges
            hoverinfo='none',
        )
    ))
    # highlight_indices = [i for i, c in enumerate(link_colors) if not c.startswith('rgba(128,128,128,')] 
    # # highlight_links = []  # example: red links

    # overlay_sources = [sources[i] for i in highlight_indices]
    # overlay_targets = [targets[i] for i in highlight_indices]
    # overlay_values  = [values[i] for i in highlight_indices]
    # overlay_colors  = [link_colors[i] for i in highlight_indices]

    # highlight_nodes = [nodes[i] for i in set(overlay_sources + overlay_targets)]
    # print(nodes)

    # fig.add_trace(go.Sankey(
    #     arrangement="fixed",
    #     node=dict(
    #         pad=15,
    #         thickness=15,
    #         color='rgba(0,0,0,0)',  # transparent node borders
    #         x=[n["x"] for n in highlight_nodes],
    #         y=[n["y"] for n in highlight_nodes],
    #         line=dict(color='rgba(0,0,0,0)', width=0),  # transparent node borders
    #         label=[""]*len(highlight_nodes),
    #     ),
    #     link=dict(
    #         source=overlay_sources,
    #         target=overlay_targets,
    #         value=overlay_values,
    #         color= overlay_colors,
    #         line=dict(color='purple', width=3),
    #     )
    # ))
    return fig
    
def compute_incoming_flows(data):
    """Returns a dict mapping each node to a list of (source, value) tuples."""
    incoming = defaultdict(list)
    for cue, r1, r2 in data:
        val = data[(cue, r1, r2)]
        incoming[r1].append((cue, val))
        incoming[r2].append((r1, val))
    return incoming

def compute_outgoing_flows(data):
    """Returns a dict mapping each node to a list of (target, value) tuples."""
    outgoing = defaultdict(list)
    for cue, r1, r2 in data:
        val = data[(cue, r1, r2)]
        outgoing[cue].append((r1, val))
        # outgoing[cue].append((r2, val))
    return outgoing

def add_vertical_bars(fig, nodes, in_or_out_flows, data, highlight_data, N, HIGHLIGHT_COLOR,
                      post_cue_thr, bef_R1_thr, bef_R2_thr, bar_width=0.03, gap=0.00):
    """Draws vertical bars next to nodes to show incoming link proportions."""
    node_lookup = {node["id"]: node for node in nodes}

    for node_id, flows in in_or_out_flows.items():
        location_idx = 0 # draw at cue location
        thr = post_cue_thr
        if node_id in ('R1', 'notR1'):
            location_idx = 1
            thr = bef_R1_thr
        elif node_id in ('R2', 'notR2'):
            location_idx = 2
            thr = bef_R2_thr
        node = node_lookup[node_id]
        x0 = node["x"] + (-0.08 if location_idx!=0 else 0.08)
        y_center = 1-node["y"]  # Plotly's y=0 at bottom
        bar_total_height = node["height"]*.85
        y_cursor = y_center + 0.5 * bar_total_height  # start from top
    
        for source, val in flows:
            h = (val/N) *.85 # scale down the bar a bit
            
            if val == 0:
                continue

            # now it gets ugly
            if node_id.endswith("R1") or node_id.startswith("Cue"): # location 1
                # we don't know which exact R2 choice this relates to, check which value matches
                # here source is the cue1 vs cue2, and node_id is the R1 or notR1
                if node_id.startswith("Cue"): # 
                    cue, r1 = node_id, source # cue node
                if node_id.endswith("R1"): # 
                    r1, cue = node_id, source # R1 node
                r2_info = data[cue, r1][data[cue, r1] == val].index
                if len(r2_info) == 0:
                    print(f"Warning: No matching R2 choice for {node_id} <- {source} with value {val}")
                    continue
                print(val)
                activation = highlight_data.loc[(cue, r1, r2_info[0]), location_idx].item()
            
            elif node_id.endswith("R2"): # location 2
                cue_info = data.index.get_level_values(0) # simple, one one 
                activation = highlight_data.loc[(cue_info, source, node_id), location_idx].item()
            
            # print(activation)
            # colormap from 0 to 1, for activation values .6, to 1.4

            print(activation, ">", thr, activation > thr)
            if activation > thr:
                color = activation_to_color(activation, vmin=1.2, vmax=3)
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x0 + bar_width,
                    y0=y_cursor,
                    y1=y_cursor - h,
                    xref="paper",
                    yref="paper",
                    line=dict(color='black', width=0),
                    fillcolor=color,
                    # fillcolor=HIGHLIGHT_COLOR,
                    name=activation,
                )
                
                # fig.add_annotation(
                #         x=(x0 + x0 + bar_width) / 2,
                #         y=(y_cursor - h/2),
                #         xref="paper",
                #         yref="paper",
                #         text=f"{activation:.2f}",
                #         showarrow=False,
                #         font=dict(size=5, color='black'),
                #         align="center",
                #         # bgcolor="white",
                #         # bordercolor="blue",
                #         borderwidth=1,
                # )
                
            y_cursor -= h + gap


def add_node_annotations(fig, nodes):
    """Adds manual annotations instead of default node labels."""
    # Remove Plotly's native labels
    fig.update_traces(node=dict(label=[""] * len(nodes)))

    for node in nodes:
        label = node["label"]
        x = node["x"]
        y = 1-node["y"]

        if node["id"].startswith("Cue"):
            pass
        else:
            # Left of bar
            fig.add_annotation(
                x=x - 0.08,
                y=y,
                text=label,
                showarrow=False,
                font=dict(size=11, color='black'),
                xanchor='right',
                yanchor='middle'
            )


def extract_trial_numbers(encoding_data):
    trial_d = encoding_data[~encoding_data.trial_id.duplicated()]

    cue_1_data  = {
        ('Cue1', 'R1', 'R2'): ((trial_d.cue ==1) & (trial_d.choice_R1 == 1) & (trial_d.choice_R2 == 1)).sum().item(),
        ('Cue1', 'R1', 'notR2'): ((trial_d.cue ==1) & (trial_d.choice_R1 == 1) & (trial_d.choice_R2 == 0)).sum().item(),
        ('Cue1', 'notR1', 'R2'): ((trial_d.cue ==1) & (trial_d.choice_R1 == 0) & (trial_d.choice_R2 == 1)).sum().item(),
        ('Cue1', 'notR1', 'notR2'): ((trial_d.cue ==1) & (trial_d.choice_R1 == 0) & (trial_d.choice_R2 == 0)).sum().item(),
    }
    cue_2_data  = {
        ('Cue2', 'R1', 'R2'): ((trial_d.cue ==2) & (trial_d.choice_R1 == 1) & (trial_d.choice_R2 == 1)).sum().item(),
        ('Cue2', 'R1', 'notR2'): ((trial_d.cue ==2) & (trial_d.choice_R1 == 1) & (trial_d.choice_R2 == 0)).sum().item(),
        ('Cue2', 'notR1', 'R2'): ((trial_d.cue ==2) & (trial_d.choice_R1 == 0) & (trial_d.choice_R2 == 1)).sum().item(),
        ('Cue2', 'notR1', 'notR2'): ((trial_d.cue ==2) & (trial_d.choice_R1 == 0) & (trial_d.choice_R2 == 0)).sum().item(),
    }
    return cue_1_data, cue_2_data

def extract_ens_activations(encoding_data, ens_selection, window):
    cue1_highlight_data = {}
    cue2_highlight_data = {}
    for which_cue in ['Cue1', 'Cue2']:
        for r1_choice in ['R1', 'notR1']:
            for r2_choice in ['R2', 'notR2']:
                trials_dat = encoding_data[(encoding_data.cue == (1 if which_cue == 'Cue1' else 2)) &
                                    (encoding_data.choice_R1 == (1 if r1_choice == 'R1' else 0)) &
                                    (encoding_data.choice_R2 == (1 if r2_choice == 'R2' else 0))]
                if trials_dat.empty:
                    continue
                
                activation = []
                for i, event_name in enumerate(['enter_afterCueZone', 'enter_reward1Zone', 'enter_reward2Zone']):
                    dat = trials_dat[trials_dat.t0_event_name == event_name][ens_selection].unstack(level='interval_t')
                    dat = dat.iloc[:, window[0]:window[1]]
                    dat = dat.T.rolling(window=6, center=True, min_periods=6).median().dropna(how='all').T
                    activation.append(dat.mean().mean())
                
                if which_cue == 'Cue1':
                    cue1_highlight_data[(which_cue, r1_choice, r2_choice)] = activation
                elif which_cue == 'Cue2':
                    cue2_highlight_data[(which_cue, r1_choice, r2_choice)] = activation
    return cue1_highlight_data, cue2_highlight_data

def render_plot(encoding_data, ens_selection, which_cue='Cue1',
                post_cue_thr=0, bef_R1_thr=0, bef_R2_thr=0,
                window=(20,30)):
    # Metadata for each unique node
    NODE_META = {
        'R1':     dict(label='stop',  color='darkgrey',          x=0.4),
        'notR1':  dict(label='skip',  color='rgb(255,255,255)',  x=0.4),
        'R2':     dict(label='stop',  color='lightgrey',         x=0.8),
        'notR2':  dict(label='skip',  color='rgb(255,255,255)',  x=0.8),
    }
    
    cue1_data, cue2_data = extract_trial_numbers(encoding_data)
    cue1_highlight, cue2_highlight = extract_ens_activations(encoding_data, ens_selection, window)
    
    if which_cue == 'Cue1':
        DATA = cue1_data
        HIGHLIGHT_DATA = cue1_highlight
        NODE_META.update({
            'Cue1':   dict(label='Cue1',  color='orange',          x=0.001),
        })
    elif which_cue == 'Cue2':
        DATA = cue2_data
        HIGHLIGHT_DATA = cue2_highlight
        NODE_META.update({
            'Cue2':   dict(label='Cue2',  color='purple',            x=0.001),
        })


    WIN_COLOR = 'rgba(79,212,79, .15)'  # light green
    LOOSE_COLOR = 'rgba(214,14,14, 0.15)'  # light red
    HIGHLIGHT_COLOR = 'rgba(0,208,255, 0.7)'

    # === EXECUTION ===


    color_outcome = True
    px_per_trial = 4
    N = sum(DATA.values())

    node_values = compute_node_values(DATA)
    columns = organize_nodes_by_column(NODE_META, node_values)
    final_nodes = compute_y_positions(columns, NODE_META)
    label_index = {node["id"]: idx for idx, node in enumerate(final_nodes)}
    sources, targets, values, link_colors = build_links(DATA, label_index, color_outcome, WIN_COLOR, LOOSE_COLOR)

    fig = build_sankey_figure(final_nodes, sources, targets, values, link_colors)
    
    add_node_annotations(fig, final_nodes)
    
    data = pd.Series(DATA)
    highlight_data = pd.DataFrame(HIGHLIGHT_DATA).T
    add_vertical_bars(fig, final_nodes, compute_incoming_flows(DATA), data, highlight_data, N, HIGHLIGHT_COLOR,
                      post_cue_thr, bef_R1_thr, bef_R2_thr)
    add_vertical_bars(fig, final_nodes, compute_outgoing_flows(DATA), data, highlight_data, N, HIGHLIGHT_COLOR,
                      post_cue_thr, bef_R1_thr, bef_R2_thr)

    sessions = _get_session_ids(encoding_data).tolist()
    if len(sessions) > 1:
        sess_str = f"{_format_session_id(sessions[0])}-{_format_session_id(sessions[-1])}"
    elif len(sessions) == 1:
        sess_str = _format_session_id(sessions[0])
    else:
        sess_str = "NA"
    fig.update_layout(
        title_text=f"{ens_selection} S{sess_str}",
        font_size=8,
        height=N*px_per_trial,
        width=250,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    # fig.show()
    return fig
