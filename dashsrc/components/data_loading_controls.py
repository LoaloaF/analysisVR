from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc

import pandas as pd

from data_loading.animal_loading import get_animal_modality
from analysis_core import NAS_DIR
from .constants import *

from data_loading.paradigm_loading import get_paradigm_modality

def render(app: Dash, data: pd.DataFrame) -> html.Div:
    
    @app.callback(
        Output(D2M_LOAD_DATA_BUTTON_ID, 'style'),
        Output('data-loaded', 'data'),
        Input(D2M_LOAD_DATA_BUTTON_ID, 'n_clicks'),
        State(D2M_ANIMAL_DROPDOWN_ID, 'value'),
        State(D2M_MODALITY_DROPDOWN_ID, 'value'),
        State(D2M_CHECKLIST_ID, 'value'),
    )
    def load_data(n_clicks, selected_animals, selected_modalities, checklist):
        if n_clicks and selected_modalities and selected_animals:
            _load_all_data(selected_animals, selected_modalities, checklist, data)
            return {"marginTop": 20, "backgroundColor": "green"}, True
        return {"marginTop": 20, "backgroundColor": "blue"}, False
    
    return dbc.Col([
        dbc.Row([
            dbc.Col([
                html.H5("Data2Memory", style={"marginTop": 20}),
                dcc.Dropdown(
                    id=D2M_ANIMAL_DROPDOWN_ID,
                    options=[{'label':f"rYL_{a:03}", 'value':a} for a in ANIMALS],
                    value=[8],
                    multi=True,
                    placeholder="Select one or more animals"
                ),
                dcc.Dropdown(
                    id=D2M_MODALITY_DROPDOWN_ID,
                    options= [{'label': modality, 'value': modality} for modality in MODALITIES],
                    multi=True,
                    value=['unity_frame'],
                    placeholder="Select one or more modalities"
                ),
                html.Div(style={"marginTop": 20})
                
            ], width=6, ),  # Light blue background for debugging
            dbc.Col([
                dcc.Checklist(
                    options=[
                        {'label': ' fromCache', 'value': 'fromCache'},
                    ],
                    value=['fromCache'],
                    id=D2M_CHECKLIST_ID,
                    style={"marginTop": 20}
                ),
                dbc.Button("Load Data", id=D2M_LOAD_DATA_BUTTON_ID, color="primary", style={"marginTop": 5}),
            ], width=6, )  # Light red background for debugging
        ])
    ], width=6) 
    
def _load_all_data(selected_animals, selected_modalities, checklist, data):
    cache = 'from' if 'fromCache' in checklist else 'to'
    # nas_dir = get_default_nas_dir() if cache == 'to' else None
    # raise ValueError("screw you")
    for modality in selected_modalities:
        modality_data = []  
        # for animal_id in selected_animals:
            # if modality == 'unity_trial':
            #     modality_parsing_kwargs = {"columns": ['trial_id', 'cue', 'trial_outcome', 
            #                                    'trial_pc_duration',]}
            # elif modality == 'unity_frame':
            #     modality_parsing_kwargs = {"columns": ['trial_id', 'frame_pc_timestamp', 
            #                                    'frame_id', 'frame_z_position', 
            #                                    "ballvelocity_first_package", 
            #                                    "ballvelocity_last_package"]}
            # # data = get_animal_modality(paradigm_id, animal_id, "unity_frame", from_nas=nas_dir,
            # #                cache='to',
            # #                modality_parsing_kwargs={"columns": ['trial_id', 'frame_pc_timestamp', 'frame_id',
            # #                                             'frame_z_position', "ballvelocity_first_package", "ballvelocity_last_package"]},
            # #                complement_data=True, position_bin_index=True)
                           
            # # d = get_animal_modality(800, animal_id, modality, from_nas=NAS_DIR,
            # #                         cache=cache, modality_parsing_kwargs=modality_parsing_kwargs,
            # #                         complement_data=True, #to_deltaT_from_session_start=True,
            # #                         position_bin_index=True)
            # # d = get_animal_modality(800, animal_id, modality, 
            # #                         complement_data=True, position_bin_index=True)
            
            # modality_data.append(d)
        print(selected_animals, selected_modalities, cache)
        # args: skip_animals, from_date, to_date
        paradigm_parsing_kwargs={"skip_animals":[1,2,4,5,6,7,8]}
        # args: skip_sessions, from_date, to_date
        animal_parsing_kwargs={"skip_sessions":["2024-07-29_16-58_rYL001_P0800_LinearTrack_41min",
                                                "2024-07-24_17-45_rYL003_P0800_LinearTrack_25min"], # bell vel doesn't match up with unity frames
                            #   "from_date": "2024-07-29", "to_date": "2024-09-03",
                            }
        # args: to_deltaT_from_session_start pct_as_index us2s event_subset na2null
        # complement_data position_bin_index rename2oldkeys
        session_parsing_kwargs={"complement_data":True, "to_deltaT_from_session_start":True, 
                                "position_bin_index": True}
        modality_data = get_paradigm_modality(paradigm_id=800, modality=modality, cache=cache, 
                                    **paradigm_parsing_kwargs,
                                    animal_parsing_kwargs=animal_parsing_kwargs,
                                    session_parsing_kwargs=session_parsing_kwargs,
                        )
        
        data[modality] = modality_data.loc[800]
    print("in funciton")
    print(data)