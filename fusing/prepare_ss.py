import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import shutil
from fuse import FUSE
import logging
import argparse

import pandas as pd
import numpy as np

import analytics_processing.sessions_from_nas_parsing as sp
from analytics_processing.modality_loading import session_modality_from_nas
from CustomLogger import CustomLogger as Logger
from analytics_processing.analytics_constants import device_paths

from VirtualConcatFS import VirtualConcatFS


def create_virtual_concat(kwargs):
    def handle_first_session():
        traces_order = mapping['pad_id'].values
        L.logger.debug(f"Using mapping:\n{mapping}")
        # copy in the prope file for JRC from this first session
        probe_fullfname = [fn for fn in os.listdir(session_dir)
                            if fn.endswith(".prb") and not fn.startswith("._")]
        print()
        print(probe_fullfname)
        print()
        if len(probe_fullfname) != 1:
            L.logger.error(f"Expected 1 probe file for {session_dir}, found "
                                f"{len(probe_fullfname)}, {probe_fullfname}")
            exit(1)
        shutil.copyfile(os.path.join(session_dir, probe_fullfname[0]),
                        os.path.join(device_paths()[0], concat_rel_path, 'concat.prb'))
        return traces_order

    L = Logger()
    mount = kwargs.pop('mount')
    root = kwargs.pop('root')
    concat_rel_path = kwargs.pop('concat_rel_path')
    concat_mount_fullfname = kwargs.pop('concat_mount_fullfname')
    
    session_fullfnames, _ = sp.sessionlist_fullfnames_from_args(**kwargs)
    
    singlefiles_fullfnames = []
    singlefiles_lengths = {}
    traces_order = None
    for i, session_ffname in enumerate(session_fullfnames):
        session_dir = os.path.dirname(session_ffname)
        d, mapping = session_modality_from_nas(session_ffname, key='ephys_traces')
        
        # valid session, rm curated traces clause later, just not processed
        if mapping is not None and mapping.columns[-1] == 'curated_trace':
            # first valid session, parse el order that is expected of all sessions
            if traces_order is None:
                traces_order = handle_first_session()
            
            # all follwing sessions
            else:
                if not all(traces_order == mapping['pad_id'].values):
                    L.logger.warning(f"Traces order of {os.path.basename(session_ffname)}"
                                     " differ from first session. Skipping.")
                    continue
        
            ephys_fname = [fn for fn in os.listdir(session_dir) 
                           if fn.startswith('20') and fn.endswith("ephys_traces.dat")][0]

            singlefiles_fullfnames.append(os.path.join(session_dir, ephys_fname))
            singlefiles_lengths[ephys_fname] = d.shape[1]
    
    # save the lengths of the single files, perhpas needed for easy reconstruction            
    lengths_fullfname = os.path.join(device_paths()[0], concat_rel_path, 'concat_boundaries.csv')
    pd.Series(singlefiles_lengths).to_csv(lengths_fullfname, header=False)
    
    
    
    L.spacer("debug")
    L.logger.debug(L.fmtmsg(singlefiles_fullfnames))
    L.spacer("debug")
            
            
    L.logger.info(f"Mounting virtual filesystem for concatenated read of "
                  f"{len(singlefiles_fullfnames)} ephys traces files...")
    # singlefiles_fullfnames = ['/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min_393_ephys_traces.dat', 
                            #   '/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min_393_ephys_traces.dat'
    # ]    
    
    logging.basicConfig(level=logging.INFO, )
    FUSE(VirtualConcatFS(root, singlefiles_fullfnames=singlefiles_fullfnames,
                         concat_mount_fullfname=concat_mount_fullfname,), 
         mount, 
         nothreads=True, 
         foreground=True)


def main():
    nas_dir = device_paths()[0]
    
    argParser = argparse.ArgumentParser("Mount a virtual file system to read concatenated ephys traces")
    argParser.add_argument('--mount', default=f"{nas_dir}/fuse_root1")
    argParser.add_argument('--root', default=f"{nas_dir}/RUN_rYL006")
    argParser.add_argument('--concat_rel_path', default="RUN_rYL006/rYL006_concat_ss")
    argParser.add_argument('--concat_mount_fullfname', default="/rYL006_concat_ss/concat.dat")
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