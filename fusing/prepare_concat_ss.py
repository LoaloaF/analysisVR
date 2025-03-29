import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import shutil
from fuse import FUSE
import logging
import argparse

import pandas as pd

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
                            if fn.endswith(".prb") and not fn.startswith('._')]
        if len(probe_fullfname) != 1:
            L.logger.error(f"Expected 1 probe file for {session_dir}, found "
                                f"{len(probe_fullfname)}, {probe_fullfname}")
            exit(1)
        # shutil.copyfile(os.path.join(session_dir, probe_fullfname[0]),
        #                 os.path.join(device_paths()[0], concat_rel_path, 'concat.prb'))
        return traces_order, os.path.join(session_dir, probe_fullfname[0])

    def handle_path_setup(kwargs, traces_order, prb_ffname, name):
        nas_dir = device_paths()[0]
        animal_id = kwargs['animal_ids'][0]
        # everything in root will appear in the mount
        root = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}")
        concat_ss_base_dir = os.path.join(root, 'concatenated_ss')
        if not os.path.exists(concat_ss_base_dir):
            os.makedirs(concat_ss_base_dir) # created once per animal

        # define the dir name of this concatenated ss preparation, R/W in this dir
        datet = pd.to_datetime('now').strftime("%Y-%m-%d_%H-%M")
        this_ss_dirname = f"{datet}_rYL{animal_id:03}_{len(traces_order)}_concat_ss{name}"
        this_ss_fullpath = os.path.join(concat_ss_base_dir, this_ss_dirname)
        os.makedirs(this_ss_fullpath, exist_ok=True)
        L.logger.info(f"Created directory for concatenated ss: {this_ss_fullpath}")

        # create the concat files that will be read by the FUSE
        open(os.path.join(this_ss_fullpath, 'concat.dat'), 'w').close()
        shutil.copyfile(prb_ffname, os.path.join(this_ss_fullpath, 'concat.prb'))
        shutil.copyfile(prb_ffname.replace(".prb", ".xml"), 
                        os.path.join(this_ss_fullpath, 'concat.xml'))
        return this_ss_dirname, root

    L = Logger()
    # extract the arguments for paths for later
    nas_dir = device_paths()[0]
    mount = os.path.join(nas_dir, kwargs.pop('mount'))
    name = kwargs.pop('name')
    assert len(kwargs['animal_ids']) == 1, "Traces from only one animal can be concatenated"
    
    # get the matching sessions
    session_fullfnames, _ = sp.sessionlist_fullfnames_from_args(**kwargs)
    
    singlefiles_fullfnames = []
    # singlefiles_lengths = {}
    traces_order, prb_ffname = None, None # set at first session
    for session_ffname in session_fullfnames:
        session_dir = os.path.dirname(session_ffname)
        d, mapping = session_modality_from_nas(session_ffname, key='ephys_traces')
        
        # valid session, rm curated traces clause later, just not processed
        if mapping is not None and mapping.columns[-1] == 'curated_trace':
            # first valid session, parse el order that is expected of all sessions
            if traces_order is None:
                traces_order, prb_ffname = handle_first_session()
            
            # all follwing sessions
            else:
                # traces must exactly match the first session
                if (len(traces_order) != len(mapping) or 
                    not all(traces_order == mapping['pad_id'].values)):
                    L.logger.warning(f"Traces order of {os.path.basename(session_ffname)}"
                                     " differ from first session. Skipping.")
                    continue
            ephys_fname = [fn for fn in os.listdir(session_dir) 
                           if fn.startswith('20') and fn.endswith("ephys_traces.dat")][0]
            singlefiles_fullfnames.append(os.path.join(session_dir, ephys_fname))
            # singlefiles_lengths[ephys_fname] = d.shape[1]
    
    if len(singlefiles_fullfnames) == 0:
        L.logger.error(f"No valid sessions found. Not mounting FUSE filesystem.")
        return
    L.spacer("debug")
    L.logger.debug(L.fmtmsg(["Files:", *singlefiles_fullfnames]))
    L.spacer("debug")
    
    this_ss_dirname, root = handle_path_setup(kwargs, traces_order, prb_ffname, name)

    L.logger.info(f"Mounting virtual filesystem for concatenated read of "
                  f"{len(singlefiles_fullfnames)} ephys traces from "
                  f"{os.path.join(mount, 'concatenated_ss', this_ss_dirname,)}")
    L.spacer("info")
    logging.basicConfig(level=logging.INFO, )
    fuse_concat_fullfname = os.path.join('/', 'concatenated_ss', this_ss_dirname, 'concat.dat')
    FUSE(VirtualConcatFS(root, singlefiles_fullfnames=singlefiles_fullfnames,
                         concat_mount_fullfname=fuse_concat_fullfname,), 
         mount, 
         nothreads=True, 
         foreground=True)

def main():
    argParser = argparse.ArgumentParser("Mount a virtual file system to read concatenated ephys traces")
    argParser.add_argument("--animal_ids", nargs='+', type=int, required=True)
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--session_names", nargs='+', default=None)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    argParser.add_argument('--name', default="")
    argParser.add_argument('--mount', default="fuse_root")

    kwargs = vars(argParser.parse_args())
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info(f"Mounting virtual FS to read concatenated ephys traces")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    create_virtual_concat(kwargs)
    
if __name__ == '__main__':
    main()