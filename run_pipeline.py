# !/usr/bin/env python3 
# executable - set up paths for import
import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

import argparse

from CustomLogger import CustomLogger as Logger
import analytics_processing.sessions_from_nas_parsing as sp

from analytics_processing.analytics import get_analytics

try:
    from ephys_preprocessing.postproc_mea1k_ephys import postprocess_ephys
except ImportError as e:
    print(f"Warning: could not import ephys related module: {e}")

try:
    from general_processing.camera_helpers import hdf5_frames2mp4_gpu
    from general_processing.camera_helpers import link_dlc_training_data
except ImportError as e:
    print(f"Warning: could not import camera/computer vision related module: {e}")


# =============================================================================
# Pipeline handlers
# =============================================================================

def run_render_videos(sessionlist_fullfnames, camera_names=None, **_):
    for fullfname in sessionlist_fullfnames:
        hdf5_frames2mp4_gpu(fullfname, camera_names=camera_names)

def run_prepare_dlc_training(sessionlist_fullfnames, link_to_dir, cam_name, **_):
    for fullfname in sessionlist_fullfnames:
        link_dlc_training_data(fullfname, link_to_dir, cam_name)

def run_decompress_ephys(sessionlist_fullfnames, exclude_shanks, **_):
    postprocess_ephys(sessionlist_fullfnames=sessionlist_fullfnames, exclude_shanks=exclude_shanks)



# Pipeline registry: name -> (handler, list of extra args to extract)
PIPELINES = {
    'render_videos':        (run_render_videos, ['camera_names']),
    'prepare_dlc_training': (run_prepare_dlc_training, ['link_to_dir', 'cam_name']),
    'decompress_ephys':     (run_decompress_ephys, ['exclude_shanks']),
}


def main():
    argParser = argparse.ArgumentParser("Run processing pipelines")
    
    # Required
    argParser.add_argument("pipeline", type=str, help="Pipeline or analytic name")
    
    # Session selection (general)
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--session_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--session_names", nargs='+', default=None)
    argParser.add_argument("--excl_session_names", nargs='+', default=None)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    
    # Pipeline-specific
    argParser.add_argument("--mode", default="compute", type=str)           # analytics
    argParser.add_argument("--exclude_shanks", nargs='+', default=None, type=int)  # ephys
    argParser.add_argument("--link_to_dir", type=str, default=None)         # dlc training
    argParser.add_argument("--cam_name", type=str, default='facecam')            # dlc training
    argParser.add_argument("--camera_names", nargs='+', default=None)         # render videos
    
    kwargs = vars(argParser.parse_args())
    
    # Setup logger
    L = Logger()
    Logger().init_logger('loggy', None, kwargs.pop("logging_level"))
    pipeline = kwargs.pop("pipeline")
    L.logger.info(f"Running pipeline: '{pipeline}'")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    
    # Dispatch
    if pipeline in PIPELINES:
        # Get session list
        sessionlist_fullfnames, _ = sp.sessionlist_fullfnames_from_args(
            kwargs.pop("paradigm_ids"), kwargs.pop("animal_ids"), kwargs.pop("session_ids"),
            kwargs.pop("session_names"), kwargs.pop("excl_session_names"),
            kwargs.pop("from_date"), kwargs.pop("to_date")
        )
        
        handler, extra_args = PIPELINES[pipeline]
        handler(sessionlist_fullfnames, **{k: kwargs.pop(k) for k in extra_args})
    else:
        # Default: treat as analytics
        get_analytics(analytic=pipeline, mode=kwargs.pop("mode"), 
                       paradigm_ids=kwargs.pop("paradigm_ids"),
                       animal_ids=kwargs.pop("animal_ids"),
                       session_ids=kwargs.pop("session_ids"),
                       session_names=kwargs.pop("session_names"),
                       excl_session_names=kwargs.pop("excl_session_names"),
                       from_date=kwargs.pop("from_date"),
                       to_date=kwargs.pop("to_date"),
                      )
        


if __name__ == '__main__':
    main()