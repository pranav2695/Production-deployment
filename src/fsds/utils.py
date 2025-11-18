import logging
import os
from datetime import datetime


def setup_logger(log_level="INFO", log_path=None, console_log=True):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # If log_path is provided, set up the file handler
    if log_path:
        log_dir = os.path.dirname(log_path)
        # Make sure the log directory exists
        os.makedirs(log_dir, exist_ok=True)  
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # If console logging is enabled, add the stream handler for console logs
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log(msg=""):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
