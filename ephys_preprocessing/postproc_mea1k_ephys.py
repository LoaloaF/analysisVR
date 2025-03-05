import os

from CustomLogger import CustomLogger as Logger
import analytics_processing.sessions_from_nas_parsing as sp

# from ../../ephysVR.git
from mea1k_modules.mea1k_raw_preproc import mea1k_raw2decompressed_dat_file, write_neuroscope_xml

def _handle_raw_mea1k_ephys(session_fullfname, mode=False, exclude_shanks=None):
    L = Logger()
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    session_dir = os.path.dirname(session_fullfname)
    animal_name = session_name.split("_")[2]
    
    # check if a dat file already exists
    dat_fnames = [f for f in os.listdir(session_dir) if f.endswith(".dat")]
    if any(dat_fnames):
        L.logger.debug(f"De-compressed ephys files found: {dat_fnames}")
        if mode != "recompute":
            L.logger.debug(f"Skipping...")
            return
        else:
            L.logger.debug(f"Recomputing...")
        
    
    if not os.path.exists(os.path.join(session_dir, "ephys_output.raw.h5")):
        L.logger.warning(f"No raw ephys recordings found for {session_name}")
        return

    # decompress the raw output of mea1k and convert to uV int16 .dat file
    # also, save a mapping csv file for identifying the rows in the dat file
    # mea1k_raw2decompressed_dat_file(session_dir, 'ephys_output.raw.h5', 
    #                                 session_name=session_name,
    #                                 animal_name=animal_name,
    #                                 convert2uV=True,
    #                                 subtract_dc_offset=True,
    #                                 exclude_shanks=exclude_shanks)
    
    write_neuroscope_xml(session_dir, "ephys_output.raw.h5",
                        animal_name=animal_name,
                         exclude_shanks=exclude_shanks)
    
    


def postprocess(paradigm_ids=None, animal_ids=None, mode=False,
                session_ids=None, session_names=None, 
                from_date=None, to_date=None, columns=None,exclude_shanks=None):
    L = Logger()
    
    if session_names is None:
        sessionlist_fullfnames, ids = sp.get_sessionlist_fullfnames(paradigm_ids, 
                                                                    animal_ids, session_ids,
                                                                    from_date, to_date)
    else:
        sessionlist_fullfnames, ids = sp.sessionnames2fullfnames(session_names)
    L.logger.info(f"Paradigm_ids: {paradigm_ids}, animal_ids: {animal_ids}, "
                   f"session_ids: {session_ids}, from_date: {from_date}, "
                   f"to_date: {to_date}\n\t"
                   f"Processing {len(sessionlist_fullfnames)} sessions\n")

    aggr = []
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        L.logger.info(f"Processing {identif} {os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        
        # create the dat file 
        _handle_raw_mea1k_ephys(session_fullfname, mode, exclude_shanks)