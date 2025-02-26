import os
from datetime import datetime
import numpy as np

from CustomLogger import CustomLogger as Logger
from analysis_utils import device_paths

def parse_paradigms_from_nas(nas_dir):
    # get list of something like "RUN_rYL001", in nas bas dir
    run_animal_names = [f for f in os.listdir(nas_dir) 
                        if f.startswith("RUN_") and str.isdigit(f[-3:])]
    paradigms = []
    for r_animal_name in run_animal_names:
        # extract the P part from something like "rYL001_P0100"
        paradigms.extend([int(f[-4:]) for f in os.listdir(os.path.join(nas_dir, r_animal_name)) 
                          if str.isdigit(f[-4:])])
    return sorted(list(set(paradigms)))
    
def parse_paradigm_animals_from_nas(paradigm_id, nas_dir):
    # get list of something like "RUN_rYL001", in nas bas dir
    run_animal_names = [f for f in os.listdir(nas_dir) 
                        if f.startswith("RUN_") and str.isdigit(f[-3:])]
    run_animal_names.sort()
    # filter out animals that did not do the passed paradigm
    filtered_animals = [r_animal_name for r_animal_name in run_animal_names if 
                        f"{r_animal_name[4:]}_P{paradigm_id:04}" # eg rYL001_P0100
                        in os.listdir(os.path.join(nas_dir, r_animal_name))]
    return sorted([int(f[-3:]) for f in filtered_animals])

def get_sessionlist_fullfnames(paradigm_ids, animal_ids, session_ids=None,
                                from_date=None, to_date=None):
    L = Logger()
    L.logger.debug("Searching NAS for applicable sessions...")
    
    nas_dir, _, _ = device_paths()
    sessionlist_fullfnames = []
    from_date = datetime.strptime(from_date, "%Y-%m-%d") if from_date is not None else None
    to_date = datetime.strptime(to_date, "%Y-%m-%d") if to_date is not None else None

    identifier = []
    # get all paradigms if not specified
    paradigm_ids = paradigm_ids if paradigm_ids is not None else parse_paradigms_from_nas(nas_dir)
    for p_id in paradigm_ids:
        # get all animals if not specified
        animal_ids_ = animal_ids if animal_ids is not None else parse_paradigm_animals_from_nas(p_id, nas_dir)
        for animal_id in animal_ids_:
            # get all sessions for the paradigm+animal combo
            parad_animal_subdir = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}", 
                                               f"rYL{animal_id:03}_P{p_id:04d}")
            if not os.path.exists(parad_animal_subdir):
                continue # not every combination exists
            
            # get all the session dirs in the animal+paradigm subdir, should end with min
            parad_animal_session_dirs = [sd for sd in os.listdir(parad_animal_subdir) 
                                         if sd.endswith("min")]
            for s_id, session_dir in enumerate(sorted(parad_animal_session_dirs)):
                date = datetime.strptime(session_dir[:10], "%Y-%m-%d")
                if from_date is not None and date < from_date:
                    continue
                if to_date is not None and date > to_date:
                    continue
                if session_ids is not None and s_id not in session_ids:
                    continue
                
                # get the session behavior data h5 file
                session_fname = [fname for fname in os.listdir(os.path.join(parad_animal_subdir, session_dir))
                                 if fname.endswith("min.hdf5")]
                if len(session_fname) != 1:
                    L.logger.warning(f"Expected 1 session file for {session_dir}, found "
                          f"{len(session_fname)}, {session_fname} in "
                          f"{os.path.join(parad_animal_subdir, session_dir)}")
                    continue
                session_fname = session_fname[0]
                
                fullfname = os.path.join(parad_animal_subdir, session_dir, session_fname)
                sessionlist_fullfnames.append(fullfname)
                identifier.append((p_id, animal_id, s_id))
    
    unique_animals = np.unique([i[1] for i in identifier])            
    L.logger.debug(f"For paradigms {paradigm_ids}, animals {unique_animals}, "
                   f"found {len(sessionlist_fullfnames)} sessions.")
    return sessionlist_fullfnames, identifier

def extract_id_from_sessionname(session_name):
    session_name_split = session_name.split("_")
    anim_name, parad_name = session_name_split[2], session_name_split[3]
    return int(anim_name[-3:]), int(parad_name[1:]), 0

def sessionnames2fullfnames(session_names):
    Logger().logger.debug(f"Inferring NAS paths for list of session names...")
    sessionlist_fullfnames, s_ids = [], []
    for session_name in session_names:
        animal_id, paradigm_id, s_id = extract_id_from_sessionname(session_name)
        session_dir = f"RUN_rYL{animal_id:03}", f"rYL{animal_id:03}_P{paradigm_id:04d}"
        fullfname = os.path.join(device_paths()[0], *session_dir, session_name, f"{session_name}.hdf5")
        sessionlist_fullfnames.append(fullfname)
        s_ids.append((s_id))
    return sessionlist_fullfnames, s_ids