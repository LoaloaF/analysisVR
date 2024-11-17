import os
import logging
import json

class CustomLogger:
    def __init__(self):
        """
        Initialize the logger with the given name and optional log file directory.

        Args:
        - name (str): The name of the logger.
        - write_to_directory (str, optional): The directory where log files will be written.
        """

        # Refer always to the same logger - only one exists per process
        self.logger = logging.getLogger("__main__")
        self.combi_msg = ""
        
        # equivalent to Parameters class, defaults if nothing passed - shound't happen
        self.CONSOLE_LOGGING_FMT = f'%(asctime)s|%(levelname)s|%(process)s|%(module)s|%(funcName)s\n\t%(message)s'
        self.FILE_LOGGING_FMT = f'%(asctime)s|%(levelname)s|%(process)s|%(module)s|%(funcName)s\n\t%(message)s'
        self.LOGGING_DIRECTORY_RUN = './'
        self.LOGGING_LEVEL = "INFO"
        
    def init_logger(self, logging_fname, logging_dir, logging_level):
        ll = logging_level if logging_level is not None else self.LOGGING_LEVEL
        self.logger.setLevel(ll)

        # setup logging to console
        console_hdlr = self._create_console_logger()
        self.logger.addHandler(console_hdlr)

        # ensusre filenames/ paths exist
        logging_dir = logging_dir if logging_dir is not None else self.LOGGING_DIRECTORY_RUN
        logging_fname = logging_fname if logging_fname is not None else "default"
        
        # setup logging to file
        file_hdlr = self._create_logfile_handler(logging_fname, logging_dir)
        self.logger.addHandler(file_hdlr)
        
    def reset_logger(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def _create_formatter(self, fmt):
        return logging.Formatter(fmt)
    
    def _create_console_logger(self):
        """
        Set up console logging with the default formatter.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._create_formatter(self.CONSOLE_LOGGING_FMT))
        return console_handler
    
    def _create_logfile_handler(self, logging_fname, write_to_directory):
        """
        Set up file-based logging with the default file formatter.

        Args:
        - write_to_directory (str): The directory where log files will be written.
        """
        try:
            log_fullfname = os.path.join(write_to_directory, f"{logging_fname}.log")
            
            file_handler = logging.FileHandler(log_fullfname)
            file_handler.setFormatter(self._create_formatter(self.FILE_LOGGING_FMT))
            return file_handler
        
        except FileNotFoundError as e:
            print(f"{e} Won't log to file.")

    def _switch_spacer_fmt(self, spacer_length):
        """
        Toggle between default formatter and spacer formatter for all handlers.
        """

        is_default = self.logger.handlers[0].formatter._fmt == self.CONSOLE_LOGGING_FMT
        if is_default:
            spacer_format = f"%(message)s{'=' * spacer_length}\n"
            spacer_fmtr = self._create_formatter(spacer_format)
            [han.setFormatter(spacer_fmtr) for han in self.logger.handlers]
        else:
            fmts = self.CONSOLE_LOGGING_FMT, self.FILE_LOGGING_FMT
            [han.setFormatter(self._create_formatter(fmt)) 
             for han, fmt in zip(self.logger.handlers, fmts)]

    def spacer(self, level="info", spacer_length=80):
        """
        Insert a separator line in the log to visually separate log entries.
        """
        self._switch_spacer_fmt(spacer_length)
        if level == "info":
            self.logger.info('')  # log a separator line
        elif level == "debug":
            self.logger.debug('')  # log a separator line
        else:
            self.logger.critical('')  # log a separator line
        self._switch_spacer_fmt(spacer_length)

    def fmtmsg(self, msg):
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=2)
        if isinstance(msg, (list, tuple)):
            msg = [json.dumps(m, indent=2) if isinstance(m, dict) else str(m) 
                   for m in msg]
            msg = ("\n\t".join(msg))
        else:
            msg = msg.replace("\n", "\n\t")
        return msg

# class ColorfulFormatter(logging.Formatter):
#     COLORS = {
#         logging.DEBUG: '\033[0;36m',  # CYAN
#         logging.INFO: '\033[0;37m',  # WHITE
#         logging.WARNING: '\033[1;33m',  # YELLOW
#         logging.ERROR: '\033[1;31m',  # RED
#         logging.CRITICAL: '\033[1;41m',  # RED BACKGROUND
#     }

#     RESET = '\033[0m'

#     def format(self, record):
#         color = self.COLORS.get(record.levelno)
#         message = super().format(record)

#         if color:
#             message = f"{color}{message}{self.RESET}"

     
#         return message
