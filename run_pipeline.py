import os
import sys
# when executed as a process add parent project dir to path, needed for ephys processing
sys.path.insert(1, os.path.join(sys.path[0], '..', 'ephysVR'))

import argparse

from CustomLogger import CustomLogger as Logger

from ephys_preprocessing.postproc_mea1k_ephys import postprocess_ephys
from analytics_processing.analytics import get_analytics

def main():
    argParser = argparse.ArgumentParser("Run pipeline for ephys processing and analytics")
    argParser.add_argument("pipeline", help="which analytic or ephys post proc to run", type=str)
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--session_names", nargs='+', default=None)
    argParser.add_argument("--excl_session_names", nargs='+', default=None)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    # for analytics
    argParser.add_argument("--mode", default="compute", type=str)
    # for ephys postproc
    argParser.add_argument("--exclude_shanks", nargs='+', default=None, type=int)
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info(f"Running pipeline for `{kwargs['pipeline']}`")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    pipeline = kwargs.pop("pipeline")
    if pipeline == "decompress_ephys":
        postprocess_ephys(**kwargs)
    else:
        kwargs.pop("exclude_shanks")
        kwargs['analytic'] = pipeline
        get_analytics(**kwargs)


if __name__ == '__main__':
    main()