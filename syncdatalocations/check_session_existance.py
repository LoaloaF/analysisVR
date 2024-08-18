import os
import re
import datetime
import pandas as pd

def extract_sessions_from_dir(path):
    def extract_info(dirname):
        try:
            # with seconds
            startime = datetime.datetime.strptime(dirname[:19], '%Y-%m-%d_%H-%M-%S')
            # calculate the file size of the directory
        except ValueError:
            try:
                # without seconds
                startime = datetime.datetime.strptime(dirname[:16], '%Y-%m-%d_%H-%M')
            except ValueError:
                print("Could not extract date from: ", dirname)
                return pd.DataFrame()
        
        size = 0
        fnames = os.listdir(os.path.join(path, dirname))
        for fname in fnames:
            full_fname = os.path.join(path, dirname, fname)
            if fname.endswith('min.hdf5'):
                size = os.path.getsize(full_fname)
                break
            size += os.path.getsize(full_fname)
        session_info = pd.DataFrame({startime: {"nfiles": len(fnames), "size(GB)":size / 1e9}})
        print(session_info)
        return session_info
        
    # Extracts all sessions from a directory
    sessions_in_dir = []
    
    for dirname in os.listdir(path):
        if os.path.isdir(os.path.join(path, dirname)):
            sessions_in_dir.append(extract_info(dirname))
    print("=======")
    #TODO:
    # because of my stupid decision to strip the seconds from the processsed 
    # session names, matching is not trivial 
    # I suggest we pick the largest local session (which still 
    # has seconds in it) and strip the seconds for this one. They way we can match
    return sessions_in_dir

def get_nas_sessions(nas_base_path, paradigms=["P0800"]):
    nas_sessions = []
    print("Extracting sessions from NAS with paradigms: ", paradigms)
    
    animal_dirs = [dirname for dirname in os.listdir(nas_base_path) if dirname.startswith('RUN_')]
    for animal_dir in animal_dirs:
        print(animal_dir)
        for paradigm_dir in os.listdir(os.path.join(nas_base_path, animal_dir)):
            if paradigm_dir[-5:] in paradigms:
                print(paradigm_dir)
                animal_paradigm_dir = os.path.join(nas_base_path, animal_dir, paradigm_dir)
                nas_sessions.extend(extract_sessions_from_dir(animal_paradigm_dir))

    return pd.concat(nas_sessions, axis=1).T
    
def extract_sessions_on_db():
    db_sessions = pd.DataFrame()
    return db_sessions

def compare_sessions(session_dfs):
    for i, session_df in enumerate(session_dfs):
        session_df.columns = [col+"_"+str(i) for col in session_df.columns]
        print(session_df)
    concat_sessions = pd.concat(session_dfs, axis=1)
    print(concat_sessions)
    

def main():
    # Path to the directory containing all the sessions
    incl_params = ["P0800"]
    nas_base_path = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning"
    nas_sessions = get_nas_sessions(nas_base_path, incl_params)

    local_data_path = "/home/loaloa/homedataXPS/projects/ratvr/VirtualReality/data/"
    local_sessions = pd.concat(extract_sessions_from_dir(local_data_path), axis=1).T
    
    connect2db = None
    db_sessions = extract_sessions_on_db()
    
    compare_sessions([nas_sessions, local_sessions, db_sessions])
    
    
if __name__ == "__main__":
    main()