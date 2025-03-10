import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from fuse import FUSE
import logging
import argparse

import analytics_processing.sessions_from_nas_parsing as sp
from analytics_processing.modality_loading import session_modality_from_nas
from CustomLogger import CustomLogger as Logger

from VirtualConcatFS import VirtualConcatFS

def create_virtual_concat(kwargs):
    L = Logger()
    mount = kwargs.pop('mount')
    root = kwargs.pop('root')
    
    session_fullfnames, _ = sp.sessionlist_fullfnames_from_args(**kwargs)
    
    singlefiles_fullfnames = []
    n_traces = None
    for i, session_ffname in enumerate(session_fullfnames):
        session_dir = os.path.dirname(session_ffname)
        _, mapping = session_modality_from_nas(session_ffname, key='ephys_traces')
        
        if mapping is not None:
            if n_traces is None:
                # first valid session
                n_traces = mapping.shape[0]
        
            ephys_fname = [fn for fn in os.listdir(session_dir) 
                           if fn.startswith('20') and fn.endswith("ephys_traces.dat")][0]
            
            if mapping.shape[0] != n_traces:
                L.logger.warning(f"Ephys traces (n={mapping.shape[0]}) of {ephys_fname} differ "
                                 f"in shape from the first recording ({n_traces=}). "
                                 f"Skipping.")
            else:
                # L.logger.debug(f"Including {ephys_fname}")
                singlefiles_fullfnames.append(os.path.join(session_dir, ephys_fname))
    L.spacer("debug")
    L.logger.debug(L.fmtmsg(singlefiles_fullfnames))
    L.spacer("debug")
            
            
    L.logger.info(f"Mounting virtual filesystem for concatenated read of "
                  f"{len(singlefiles_fullfnames)} ephys traces files...")
    # singlefiles_fullfnames = ['/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min_393_ephys_traces.dat', 
                            #   '/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min_393_ephys_traces.dat'
    # ]    
    
    logging.basicConfig(level=logging.INFO, )
    FUSE(VirtualConcatFS(root, singlefiles_fullfnames=singlefiles_fullfnames,), 
         mount, 
         nothreads=True, 
         foreground=True)
    
    
def main():
    
    argParser = argparse.ArgumentParser("Mount a virtual file system to read concatenated ephys traces")
    argParser.add_argument('--mount')
    argParser.add_argument('--root')
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--session_names", nargs='+', default=None)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    # for analytics
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info(f"Mounting virtual FS to read concatenated ephys traces")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    create_virtual_concat(kwargs)
    
if __name__ == '__main__':
    main()