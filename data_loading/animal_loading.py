import os
import pandas as pd
from datetime import datetime

from data_loading.session_loading import get_session_modality

from CustomLogger import CustomLogger as Logger

from analysis_core import NAS_DIR

from analysis_utils import join_indices

def _parse_animal_sessions_from_nas(paradigm_id, animal_id):
    paradigm_dir = f"RUN_rYL{animal_id:03}/rYL{animal_id:03}_P{paradigm_id:04d}"
    # return tuples of (nas_base_dir, session_dir, session_name) for each session
    return [(NAS_DIR, os.path.join(paradigm_dir, session_name), session_name)
            for session_name in sorted(os.listdir(os.path.join(NAS_DIR, paradigm_dir)))
            if session_name.endswith("min")]

def get_animal_modality(paradigm_id, animal_id, modality, modality_parsing_kwargs={}, 
                        session_parsing_kwargs={}, **animal_parsing_kwargs):
    L = Logger()
    
    from_nas_all_sessions = _parse_animal_sessions_from_nas(paradigm_id, animal_id)
    L.logger.info(f"Loading {modality} data for animal {animal_id} from "
                  f"NAS, found {len(from_nas_all_sessions)} sessions")
    
    animal_modality_data = []
    for i, session_dir_tuple in enumerate(from_nas_all_sessions):
        # if i >3: continue # Session5 overalping index...
        L.spacer("debug")
        L.logger.debug(f"Session {i+1}, {session_dir_tuple[2]}\n"
                       f"{os.path.join(session_dir_tuple[0], session_dir_tuple[1])}")
        L.spacer("debug")
        print(f"  {i+1}/{len(from_nas_all_sessions)}...", end='\r')
        
        # explicit exclusion of session_name
        if skip_sessions := animal_parsing_kwargs.get("skip_sessions", None):
            if session_dir_tuple[2] in skip_sessions:
                L.logger.info(f"Skipping session {session_dir_tuple[2]}")
                continue
        
        # date filtering
        if "from_data" in animal_parsing_kwargs or "to_date" in animal_parsing_kwargs:
            from_date = animal_parsing_kwargs.get("from_date", "2021-01-01")
            from_date = pd.to_datetime(from_date, format="%Y-%m-%d")
            to_date = animal_parsing_kwargs.get("to_date", datetime.now().strftime("%Y-%m-%d"))
            to_date = pd.to_datetime(to_date, format="%Y-%m-%d")
            session_date = pd.to_datetime(session_dir_tuple[2][:10], format="%Y-%m-%d")
            
            if not from_date <= session_date <= to_date:
                L.logger.info(f"Skipping session {session_dir_tuple[2]}, outisde date range")
                continue
            
        session_data = get_session_modality(session_dir_tuple=session_dir_tuple, 
                                            modality=modality, 
                                            modality_parsing_kwargs=modality_parsing_kwargs,
                                            **session_parsing_kwargs,
                                            )
        if session_data is None:
            continue
        
        midx_tuples = [(animal_id, i, tr_id) for tr_id in session_data.trial_id]
        midx = pd.MultiIndex.from_tuples(midx_tuples, names=["animal_id", "session_id", "trial_id"])
        session_data.index = join_indices(midx, session_data.index)
        animal_modality_data.append(session_data)

    if animal_modality_data:
        animal_modality_data = pd.concat(animal_modality_data, axis=0)
        s = animal_modality_data.index.get_level_values("session_id").unique().values
        L.logger.info(f"Loaded {modality} data for animal {animal_id} from "
                      f"NAS:\n{animal_modality_data}\n, sessions: {s}")
        return animal_modality_data