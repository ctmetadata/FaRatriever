import os
import logging
import inspect
from datetime import datetime


def _get_line_info():
    frame = inspect.currentframe().f_back.f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_lineno


def _setup_logger(global_rank: int):
    logger = logging.getLogger(f"distill_rank{global_rank}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        f"Rank{global_rank}-%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if global_rank == 0:
        log_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-rank0.log"
        fh = logging.FileHandler(log_name, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
