"""
Description:
    This module provides logging functionality for the application by redirecting standard output
    and standard error to a logging instance. It defines a custom stream class (StreamToLogger)
    that intercepts output messages and writes them to a logger at a specified logging level.
    The module also provides a setup_logging() function to configure the logging format, file, and 
    to redirect sys.stdout and sys.stderr to the custom logger.

Key Components:
    - StreamToLogger: A class that wraps a logger to redirect stream output.
    - setup_logging: A function that sets up the logging configuration and redirects standard output
      and standard error.

Usage:
    At the start of your application, call setup_logging() to initialize logging:
        from log_redirector import setup_logging
        setup_logging(log_filename='app.log', log_level=logging.INFO)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import logging
import sys

class StreamToLogger:
    """
    Custom stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        """
        Initialize the StreamToLogger object.

        Args:
            logger (logging.Logger): The logger to which messages will be redirected.
            log_level (int, optional): The logging level for the messages (default is logging.INFO).
        """
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        """
        Write a message to the logger if it is not empty after stripping whitespace.

        Args:
            message (str): The message to write.
        """
        # Remove any trailing whitespace from the message
        if message.rstrip() != "":
            # Log the message at the specified log level
            self.logger.log(self.log_level, message.rstrip())

    def flush(self):
        """
        Flush method for compatibility. Does nothing as logging handles flushing.
        """
        pass

def setup_logging(log_filename='app.log', log_level=logging.INFO):
    """
    Set up logging configuration and redirect stdout and stderr to a log file.

    This function configures the basic logging settings including log level, format, filename,
    and file mode. It then creates a logger and redirects sys.stdout and sys.stderr to custom
    StreamToLogger instances to capture all output into the log file.

    Args:
        log_filename (str, optional): The filename for the log file (default is 'app.log').
        log_level (int, optional): The logging level (default is logging.INFO).
    """
    # Configure the basic logging settings
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='w'
    )

    # Create a logger instance with the name 'SIRLogger'
    logger = logging.getLogger('SIRLogger')
    logger.setLevel(log_level)

    # Redirect standard output and standard error to the logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
