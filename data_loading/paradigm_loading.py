import os
import pandas as pd
import hashlib

from data_loading.animal_loading import get_animal_modality

from CustomLogger import CustomLogger as Logger

from analysis_core import LOCAL_DATA_DIR, NAS_DIR

def _parse_paradigm_animals_from_nas(paradigm_id):
    # get list of something like "RUN_rYL001", in nas bas dir
    run_animal_names = [f for f in os.listdir(NAS_DIR) 
                        if f.startswith("RUN_") and str.isdigit(f[-3:])]
    run_animal_names.sort()
    # filter out animals that did not do the passed paradigm
    filtered_animals = [r_animal_name for r_animal_name in run_animal_names if 
                        f"{r_animal_name[4:]}_P{paradigm_id:04}" # eg rYL001_P0100
                        in os.listdir(os.path.join(NAS_DIR, r_animal_name))]
    # return animal ids
    return [int(f[-3:]) for f in filtered_animals]
    
def _get_cache_fullfname(identifier, kwargs):
    # Convert kwargs to a sorted string representation
    kwargs_str = "_".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    # Create a hash of the kwargs string to ensure the filename is not too long
    kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
    filename = f"{identifier}_{kwargs_hash}.pkl"
    Logger().logger.debug(f"Cache filename: {filename}")
    return os.path.join(LOCAL_DATA_DIR, filename)

def get_paradigm_modality(paradigm_id, modality, cache=None, 
                          modality_parsing_kwargs={}, session_parsing_kwargs={}, 
                          animal_parsing_kwargs={}, **paradigm_parsing_kwargs):
    L = Logger()
    
    identifier = f"P{paradigm_id:04d}_{modality}"
    kwargs = {**modality_parsing_kwargs, **session_parsing_kwargs, **animal_parsing_kwargs}
    fullfname = _get_cache_fullfname(identifier, kwargs)
    if cache == 'from':
        if not os.path.exists(fullfname):
            L.logger.error(f"Cache file not found: {fullfname} with args: {kwargs}")
            exit(1)
        paradigm_modality_data = pd.read_pickle(fullfname)

    else:
        animal_ids = _parse_paradigm_animals_from_nas(paradigm_id=paradigm_id)
        # skip animals if specified
        if skip_animals := paradigm_parsing_kwargs.get("skip_animals", None):
            animal_ids = [a_id for a_id in animal_ids if a_id not in skip_animals]

        L.spacer(), L.spacer()
        L.logger.info(f"Loading paradigm {paradigm_id} {modality} data for "
                      f"{len(animal_ids)} animals {animal_ids}")
        
        paradigm_modality_data = []
        for animal_id in animal_ids:
            animal_data = get_animal_modality(paradigm_id, animal_id, modality, 
                                              modality_parsing_kwargs=modality_parsing_kwargs,
                                            session_parsing_kwargs=session_parsing_kwargs,
                                            **animal_parsing_kwargs)
            if animal_data is not None:
                animal_data = pd.concat({paradigm_id: animal_data}, names=["paradigm_id"])
                paradigm_modality_data.append(animal_data)

        if paradigm_modality_data:
            paradigm_modality_data = pd.concat(paradigm_modality_data, axis=0)
            if cache == 'to':
                L.logger.info(f"Caching paradigm {paradigm_id} {modality} data to {fullfname}")
                paradigm_modality_data.to_pickle(fullfname)
        else:
            return
    
    animal_ids = paradigm_modality_data.index.get_level_values("animal_id").unique().values
    L.logger.info(f"Loaded paradigm {paradigm_id} {modality} data for "
                f"{len(animal_ids)} animals {animal_ids}:\n{paradigm_modality_data}")
    return paradigm_modality_data