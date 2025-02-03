import argparse

from CustomLogger import CustomLogger as Logger
from analytics_processing import analytics

def main():
    argParser = argparse.ArgumentParser("Run pipeline to calculate analytics")
    argParser.add_argument("analytic", help="which analytic to compute", type=str)
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--sessionlist_fullfnames", nargs='+', default=None)
    argParser.add_argument("--recompute", action="store_true", default=False)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info(f"Running pipeline for analytic `{kwargs['analytic']}`")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    if not kwargs.pop("recompute"):
        kwargs['mode'] = 'set'
    else:
        kwargs['mode'] = 'recompute'
    analytics.get_analytics(**kwargs)


if __name__ == '__main__':
    main()