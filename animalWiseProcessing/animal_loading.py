import os
import pandas as pd
from sessionWiseProcessing.session_loading import get_session_modality

from CustomLogger import CustomLogger as Logger

def _parse_animal_sessions_from_nas(nas_base_dir, paradigm_id, animal_id):
    paradigm_dir = f"RUN_rYL00{animal_id}/rYL{animal_id:03}_P{paradigm_id:04d}"
    # return tuples of (nas_base_dir, session_dir, session_name) for each session
    return [(nas_base_dir, os.path.join(paradigm_dir, session_name), session_name)
            for session_name in sorted(os.listdir(os.path.join(nas_base_dir, paradigm_dir)))
            if session_name.endswith("min")]

def _query_animal_sessions_from_db(db_fullfname, paradigm, animal):
    raise NotImplementedError("Querying animal sessions from database is not implemented yet")

def get_animal_modality(paradigm_id, animal_id, modality, from_nas=None, from_db=None, **kwargs):
    if modality == "metadata":
        raise NotImplementedError("Loading metadata is not implemented yet")
    
    L = Logger()
    if from_nas is not None:
        from_nas_all_sessions = _parse_animal_sessions_from_nas(from_nas, paradigm_id, animal_id)
        L.logger.info(f"Loading {modality} data for animal {animal_id} from "
                      f"NAS, found {len(from_nas_all_sessions)} sessions...")
        
        animal_modality_data = []
        for i, from_nas in enumerate(from_nas_all_sessions):
            session_data = get_session_modality(from_nas=from_nas, 
                                                modality=modality, 
                                                **kwargs)
            if session_data is None:
                continue
            midx = pd.MultiIndex.from_tuples([(animal_id, i, ID_) for ID_ in session_data.trial_id])
            session_data.index = midx.set_names(["animal_id", "session_id", "trial_id"])
            animal_modality_data.append(session_data)

    elif from_db is not None:
        animal_session_names = _query_animal_sessions_from_db(*from_db)
    else:
        raise ValueError("Either from_nas or from_db must be provided")

    if animal_modality_data:
        animal_modality_data = pd.concat(animal_modality_data, axis=0)
        L.logger.info(f"Loaded {modality} data for animal {animal_id} from NAS:\n{animal_modality_data}")
        return animal_modality_data