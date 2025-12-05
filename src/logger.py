import os
import sys
import logging

def set_logger(logger_name: str, level=logging.INFO):
    """Get a logger instance."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logger_name + '.log')
    logging.basicConfig(
            filename=log_path,
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode='a'
        )

    logger = logging.getLogger(logger_name)
    logger.addHandler(logging.StreamHandler(sys.stdout)) # Also log to console
    logger.setLevel(level)
    return logger

logger = set_logger("robotdog_logger")