import argparse
import os
import sys

from loguru import logger


def create_logger(args: argparse.Namespace):
    """Create logger for flask app.
    Args:
        app (Flask): Flask app.
    """
    logger.remove()

    log_dir = args.log_dir
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)

    logger_config = {
        "handlers": [{
            "sink": sys.stdout,
            "level": "INFO"
        }, {
            "sink": f"{log_dir}/info.log",
            "backtrace": False,
            "diagnose": False,
            "rotation": "00:00"
        }, {
            "sink": f"{log_dir}/error.log",
            "backtrace": True,
            "diagnose": True,
            "filter": lambda record: record["level"].name == "ERROR"
        }]
    }
    logger.configure(**logger_config)