import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))

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

from mea1k_modules.mea1k_raw_preproc import _write_prb_file

def create_virtual_concat(kwargs):
    def handle_path_setup(animal_id, traces_order, name):
        nas_dir = device_paths()[0]
        # everything in root will appear in the mount
        concat_ss_base_dir = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}", 'concatenated_ss')
        if not os.path.exists(concat_ss_base_dir):
            os.makedirs(concat_ss_base_dir) # created once per animal

        # define the dir name of this concatenated ss preparation, R/W in this dir
        datet = pd.to_datetime('now').strftime("%Y-%m-%d_%H-%M")
        this_ss_dirname = f"{datet}_rYL{animal_id:03}_{len(traces_order)}_concat_ss{name}"
        this_ss_fullpath = os.path.join(concat_ss_base_dir, this_ss_dirname)
        os.makedirs(this_ss_fullpath, exist_ok=True)
        L.logger.info(f"Created directory for concatenated ss: {this_ss_fullpath}")
        
        # the JRC parameter file
        shutil.copy(os.path.join(device_paths()[2], 'analysisVR/assets/concat_template.prm'), 
                    os.path.join(this_ss_fullpath, 'concat.prm'))
        # the probe file
        _write_prb_file(mapping, os.path.join(this_ss_fullpath, 'concat.prb'),
                        shank_subset=shank_subset)
        # create the concat files that will be read by the FUSE
        open(os.path.join(this_ss_fullpath, 'concat.dat'), 'w').close()
        return this_ss_dirname, this_ss_fullpath

    L = Logger()
    # extract the arguments for paths for later
    nas_dir = device_paths()[0]
    mount = os.path.join(nas_dir, kwargs.pop('mount'))
    name = kwargs.pop('name')
    shank_subset = kwargs.pop('shank_subset', None)
    assert len(kwargs['animal_ids']) == 1, "Traces from only one animal can be concatenated"
    
    # get the matching sessions
    session_fullfnames, _ = sp.sessionlist_fullfnames_from_args(**kwargs)
    
    singlefiles_fullfnames = []
    singlefiles_lengths = []
    traces_order = None # set at first session
    for session_ffname in session_fullfnames:
        session_dir = os.path.dirname(session_ffname)
        d, mapping = session_modality_from_nas(session_ffname, key='ephys_traces')
        
        # valid session
        if mapping is not None:
            # first valid session, parse el order that is expected of all sessions
            if traces_order is None:
                L.logger.debug(f"Using mapping:\n{mapping}")
                traces_order = mapping['pad_id'].values
                # setup the paths, create base dir etc
                this_ss_dirname, this_ss_fullpath = handle_path_setup(kwargs['animal_ids'][0], 
                                                                      traces_order, name)
                
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
            singlefiles_lengths.append((ephys_fname, d.shape[1]))

    if len(singlefiles_fullfnames) == 0:
        L.logger.error(f"No valid sessions found. Not mounting FUSE filesystem.")
        return
    L.spacer("debug")
    L.logger.debug(L.fmtmsg(["Files:", *singlefiles_fullfnames]))
    L.spacer("debug")
    
    # overview of the session boundaries
    singlefiles_lengths = pd.DataFrame(singlefiles_lengths, columns=['name','nsamples'])
    singlefiles_lengths['length_sec'] = singlefiles_lengths['nsamples'] // 20000
    singlefiles_lengths['length_sec_cum'] = singlefiles_lengths['length_sec'].cumsum()
    singlefiles_lengths['nsamples_cum'] = singlefiles_lengths['nsamples'].cumsum()
    
    singlefiles_lengths.to_csv(os.path.join(this_ss_fullpath, 'concat_session_lengths.csv'), index=False)

    L.logger.info(f"Mounting virtual filesystem for concatenated read of "
                  f"{len(singlefiles_fullfnames)} ephys traces at {mount}\n"
                  f"Spike sorting dir: {this_ss_fullpath}")
    L.spacer("info")
    logging.basicConfig(level=logging.INFO, )
    fuse_concat_fullfname = os.path.join('/', 'concatenated_ss', this_ss_dirname, 'concat.dat')
    FUSE(VirtualConcatFS(this_ss_fullpath, singlefiles_fullfnames=singlefiles_fullfnames,
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
    argParser.add_argument("--shank_subset", default=None, type=int, nargs='+')
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