import logging
import os
from datetime import datetime


def setup_logger(mode: str, model_name: str):

    os.makedirs("results/logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"results/logs/{mode}_{timestamp}.log"

    logger = logging.getLogger("sentiment_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info("Run started")
    logger.info(f"Mode: {mode}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Log file: {log_file}")

    return logger, log_file
