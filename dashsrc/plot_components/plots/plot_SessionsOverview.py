import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import dashsrc.components.dashvis_constants as C

def format_data(metadata):
    data = metadata.copy()
    start_time_dt = pd.DatetimeIndex(pd.to_datetime(data['start_time'], format="%Y-%m-%d_%H-%M"))
    data.loc[:, "session_start"] = start_time_dt
    data.loc[:, "session_length_min"] = pd.to_timedelta(data['duration_minutes'], unit='m')
    data.loc[:, "session_length_min"] = data.loc[:, "session_length_min"].fillna(pd.to_timedelta(15, unit='m'))
    data.loc[:, "session_stop"] = start_time_dt + data['session_length_min']
    # data.sort_index(modality=('animal_id', 'session_id'), inplace=True)
    data.sort_values(by='session_start', inplace=True)
    return data

def setup_yaxis(fig, from_hour, to_hour, data):
    # setup y axis
    yticks, ytick_labels = [], []
    from_seconds = 3600 * from_hour
    to_seconds = 3600 * to_hour
    interval_length = to_seconds-from_seconds
    first_day = data['session_start'].sort_values().iloc[0].normalize()
    last_day = data['session_stop'].sort_values().iloc[-1].normalize()
    first_ytick, last_ytick = 3600 * 16, 3600 * 20 # 12pm, 8pm
    animals = data.index.get_level_values("animal_id").unique().tolist()
    for i, animal in enumerate(animals):
        yticks.extend([yt+i*interval_length for yt in [first_ytick, last_ytick]])
        ytick_labels.extend(["4pm", "8pm"])
        
        # add animal label as manual text
        fig.add_annotation(
            x=first_day + pd.Timedelta(days=3),
            y=yticks[-1] - (yticks[-1] - yticks[-2]) / 2,
            text=f"Rat{animal:02}",
            font=dict(size=18, weight='bold'),
            showarrow=False,
        )
        
    for yt in yticks:
        fig.add_shape(
            type="line",
            x0=first_day - pd.Timedelta(days=3),
            y0=yt,
            x1=last_day + pd.Timedelta(days=3),
            y1=yt,
            line=dict(color='black', width=.5)
        )
        
    fig.update_yaxes(
        tickvals=yticks,
        ticktext=ytick_labels,
        # gridcolor='#d3d3d3',
        # gridwidth=1,
        range=(yticks[-1], from_seconds),
        ticks="outside"
    )
    return interval_length

def draw_seesions(fig, data, interval_length):
    animals = data.index.get_level_values("animal_id").unique().tolist()
    # iterate sessions metadata
    for idx, row in data.iterrows():
        parad, anim = idx[0], idx[1]
        
        # Extract day and time
        day = row['session_start'].normalize()  # Start of the day (datetime.date)
        start_time = row['session_start'].time()  # Time of day
        stop_time = row['session_stop'].time()  # Time of day
        
        # Convert time to seconds since start of day for plotting
        start_seconds = pd.to_timedelta(start_time.hour, unit='h') + pd.to_timedelta(start_time.minute, unit='m')
        stop_seconds = pd.to_timedelta(stop_time.hour, unit='h') + pd.to_timedelta(stop_time.minute, unit='m')
        
        # Calculate y-axis position for this animal
        y_animal_offset = animals.index(anim) * interval_length
        y_start = start_seconds.total_seconds() + y_animal_offset
        y_stop = stop_seconds.total_seconds() + y_animal_offset
        
        # split notes into lines that are 90 characters long
        if row.isna()['notes']:
            annot = ''
        else:
            notes = "<br>".join([row['notes'][i:i+70] for i in range(0, len(row['notes']), 70)])
            annot = (f"{row['session_start'].strftime('%d.%m, %A')}<br>"
                    f"{row['session_start'].strftime('%H:%M:%S')}<br>",
                    f"{notes}<br>",
                    )
        # Plot the session as a vertical line spanning start to stop
        fig.add_trace(
            go.Scatter(
                x=[day, day],
                y=[y_start, y_stop],
                mode='lines',
                line=dict(color=C.PARADIGM_COLORS[parad], width=6),
                showlegend=False,
                hovertext=annot,    
                hoverinfo='text'
            )
        )

def draw_modality_heatmap(data, modalities, fig):
    session_idx = np.arange(len(data))
    grid_color = np.ones((len(modalities), len(session_idx))) * -0.25
    grid_annot = np.empty((len(modalities), len(session_idx)), dtype=object)

    # iterate over modalities
    for i, modality in enumerate(modalities):
        # get the columns that contain the modality
        cols = [modality in col for col in data.columns]
        
        # iterate over sessions, checking their modality data
        for j in session_idx:
            session_modality_data = data.iloc[j, cols]
            annot = ''
            
            # if all columns are NaN, the modality is missing
            if session_modality_data.isna().all():
                color = -0.25
                annot = f'{modality} missing on NAS'
            
            else:
                # header
                annot = f'<span style="font-size:20px"><b>{modality}</b>:</span><br>'
                annot += f'<u>{data.iloc[j]["session_name"]}</u><br>'
                key = modality
                color = 2 # default color, green, make yellow or red if problems are found
                
                # check for missing frames in camera modalities
                if modality.endswith('cam'):
                    key = f"{modality}_packages"
                    n_missing_frames = session_modality_data[f"{key}_missing_frame_keys"].count(",") 
                    if n_missing_frames > 0:
                        annot += f"<span style='color:yellow'>Missing Frames: {n_missing_frames:,}</span><br>"
                        color = 1
                        
                elif modality == 'ephys_traces':
                    if not pd.isna((session_modality_data[f"{key}_n_columns"])):
                        color = 2
                    else:
                        if not session_modality_data[f"{key}_recorded"]:
                            annot += f'Compressed: False<br>'
                            color = -.25 # white
                        else:
                            annot += f'Compressed: True<br>'
                            color = 1
                    
                # get shape of the modality h5 file
                annot += f'Columns: {session_modality_data[f"{key}_n_columns"]}<br>'
                annot += f'Rows: {session_modality_data[f"{key}_n_rows"]:,}<br>'
                
                # column names of the modality h5 file
                # cols_string = 
                # print(f"{key}_columns")
                # print(cols_string)
                # print(type(cols_string))
                if f"{key}_columns" in session_modality_data and not pd.isna(session_modality_data[f"{key}_columns"]):
                    cols_string = session_modality_data[f"{key}_columns"]
                    gaps = [i for i,s in enumerate(cols_string) if s==" "][2::3]
                    for g in gaps:
                        cols_string = cols_string[:g] + "<br>   " + cols_string[g:]
                    annot += f'Columns: {cols_string}<br>'
                
                if modality not in ('metadata', 'paradigm_variable', 'ephys_traces'):
                    # check for NaN ephys timestamps
                    nan_ephys_tstamps = session_modality_data[f"{key}_n_nan_ephys_timestamp"]
                    if not pd.isna(nan_ephys_tstamps):
                        et_string = f'NaN ephys t-stamps: {nan_ephys_tstamps:,}<br>'
                        # if all timestamps are NaN, its bc of no ephys file, not unexpected
                        if nan_ephys_tstamps > 0 and nan_ephys_tstamps != session_modality_data[f"{key}_n_rows"]:
                            et_string = f"<span style='color:yellow'>{et_string}</span>"
                            color = -1 # if subset of ephys timestamps are nan, red
                        annot += et_string
                        
                    # check for unpatched ephys timestamps
                    npatched = session_modality_data[f"{key}_n_patched_ephys_timestamp"]
                    if not pd.isna(npatched):
                        ep_string =  f'Patched ephys t-stamps: {npatched:,}<br>'
                        if npatched > 0:
                            ep_string = f"<span style='color:yellow'>{ep_string}</span>"
                            color = -1 # if some ephys timestamps are patched, red
                        annot += ep_string
            grid_annot[i, j] = annot
            grid_color[i, j] = color
            
    fig.add_heatmap(
        x=np.arange(data.shape[0]),
        y=np.arange(len(modalities)),
        z=grid_color,
        colorscale=C.DATA_QUALITY_COLORS,
        showscale=False,
        text=grid_annot,
        zmin=-1,
        zmax=2,
        hovertemplate='%{text}<extra></extra>'
    )
    return grid_color, grid_annot

def setup_modality_plot_axes(fig, data, modalities):
    # add y axis grid
    for i in range(len(modalities)+1):
        fig.add_shape(
            type="line",
            x0=-.5,
            y0=i-.5,
            x1=len(data)-.5,
            y1=i-.5,
            line=dict(color='black', width=.5)
    )

    # add x axis grid
    for i in range(len(data)+1):
        fig.add_shape(
            type="line",
            x0=i-.5,
            y0=-.5,
            x1=i-.5,
            y1=len(modalities)-.5,
            line=dict(color='black', width=1.5)
        )

    # xgrid
    fig.update_xaxes(
        tickvals=np.arange(len(data)),
        ticktext=data['session_start'].dt.strftime('%d. %b.'),
        tickangle=45,
        tickfont=dict(size=12),
        showticklabels=True,
        showline=True,
        zeroline=True,
        ticks="outside",
        title='Session date'
    )

    # ygrid
    fig.update_yaxes(
        tickvals=np.arange(len(modalities)-1, -1, -1),
        ticktext=list(reversed(modalities)),
        showgrid=True,
        gridcolor='black',
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        showticklabels=True,
        title='Modality'
    )
    
    animal = data.index.get_level_values("animal_id").unique().tolist()[0]
    # add animal label as manual text
    fig.add_annotation(
        x=1,
        y=len(modalities)+.2,
        text=f"Rat{animal:02}",
        font=dict(size=18, weight='bold'),
        showarrow=False,
    )
    
    
def render_plot(metadata, from_hour, to_hour, from_date, to_date, plot_type):
    # post-process metadata to make it usable for time plotting
    data = format_data(metadata)    
    
    # Setup the plot
    fig = go.Figure()
    
    if plot_type == 'Timeseries':
        # setup y axis
        interval_length = setup_yaxis(fig, from_hour, to_hour, data)
        
        # draw session bars
        draw_seesions(fig, data, interval_length)
        
        # setup x axis
        fig.update_xaxes(
            range=[from_date, to_date],
        )

        
    elif plot_type == 'Modalities':
        modalities = ['ephys_traces', 'event', 'ballvelocity', 'metadata', 'unitycam', 'facecam', 
                    'bodycam', 'unity_frame', 'unity_trial', 'paradigm_variable',
                    ]
        
        draw_modality_heatmap(data, modalities, fig)
        
        setup_modality_plot_axes(fig, data, modalities)
        
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(t=30, b=30, l=30, r=30),
        height=400,
        xaxis=dict(scaleanchor="y", constrain="domain"),
    )
    
    return fig