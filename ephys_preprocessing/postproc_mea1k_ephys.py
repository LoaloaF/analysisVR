from datetime import datetime
import json
from time import sleep
import sys
import os
import argparse
from collections import OrderedDict

# when executed as a process add parent project dir to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))


import pandas as pd
import numpy as np

import analytics_processing.analytics_constants as C

from CustomLogger import CustomLogger as Logger
# from analytics_processing.modality_loading import get_complemented_session_modality
from analysis_utils import device_paths

import analytics_processing.sessions_from_nas_parsing as sp

from mea1k_modules.mea1k_raw_preproc import mea1k_raw2decompressed_dat_file

def _handle_raw_mea1k_ephys(session_fullfname, recompute=False, exclude_shanks=[]):
    L = Logger()
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    session_dir = os.path.dirname(session_fullfname)
    
    # check if a dat file already exists
    dat_fnames = [f for f in os.listdir(session_dir) if f.endswith(".dat")]
    if any(dat_fnames):
        L.logger.debug(f"De-compressed ephys files found: {dat_fnames}")
        if not recompute:
            L.logger.debug(f"Skipping...")
            return
        else:
            L.logger.debug(f"Recomputing...")
        
    
    if not os.path.exists(os.path.join(session_dir, "ephys_output.raw.h5")):
        L.logger.warning(f"No raw ephys recordings found for {session_name}")
        return

    # decompress the raw output of mea1k and convert to uV int16 .dat file
    # also, save a mapping csv file for identifying the rows in the dat file
    mea1k_raw2decompressed_dat_file(session_dir, 'ephys_output.raw.h5', 
                                    session_name=session_name,
                                    convert2uV_int16=True,
                                    subtract_dc_offset=True,
                                    exclude_shanks=exclude_shanks)


def postprocess(paradigm_ids=None, animal_ids=None, recompute=False,
                session_ids=None, sessionlist_fullfnames=None, 
                from_date=None, to_date=None, columns=None,exclude_shanks=[]):
    L = Logger()
    
    if sessionlist_fullfnames is None:
        sessionlist_fullfnames, ids = sp.get_sessionlist_fullfnames(paradigm_ids, 
                                                                    animal_ids, session_ids,
                                                                    from_date, to_date)
    else:
        ids = [sp.extract_id_from_sessionname(os.path.basename(s))
               for s in sessionlist_fullfnames]
    L.logger.debug(f"Paradigm_ids: {paradigm_ids}, animal_ids: {animal_ids}, "
                   f"session_ids: {session_ids}, from_date: {from_date}, "
                   f"to_date: {to_date}\n\t"
                   f"Processing {len(sessionlist_fullfnames)} sessions\n")

    aggr = []
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        L.logger.debug(f"Processing {identif} {os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        
        # create the dat file 
        _handle_raw_mea1k_ephys(session_fullfname, recompute, exclude_shanks)

            
            
def main():
    argParser = argparse.ArgumentParser("Run pipeline to postprocess mea1k ephys data")
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--exclude_shanks", nargs='+', default=None, type=int)
    argParser.add_argument("--sessionlist_fullfnames", nargs='+', default=None)
    argParser.add_argument("--recompute", action="store_true", default=False)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info("Postprocessing mea1k ephys data")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    postprocess(**kwargs)

if __name__ == '__main__':
    main()