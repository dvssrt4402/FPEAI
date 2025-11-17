# logger.py
import logging
import os
import sys

def get_logger(log_path="log", log_file="log.txt"):
    """
    Create a file+console logger. No side effects, no arg parsing, no utils import.
    main.py passes log_path/log_file explicitly.
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_file)

    file_h = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_h.setFormatter(logging.Formatter("%(levelname)s -> %(asctime)s: %(message)s"))
    file_h.setLevel(logging.INFO)

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setFormatter(logging.Formatter("%(levelname)s -> %(asctime)s: %(message)s"))
    console_h.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[file_h, console_h], force=True)
    return logging.getLogger(__name__)

if __name__ == "__main__":
    # quick self-test
    logger = get_logger()
    logger.info("test")
