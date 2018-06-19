import logging
import sys
sys.path.append("..")

import numpy as np

from utils.config import opt


def initialize_logger(file_name, debug=False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        log_format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(opt.log_path / f'{file_name}.log'),
            logging.StreamHandler()
        ])


def log_model(epoch, lr, trn_metrics, trn_time, val_metrics=None, val_time=None):
    logging.info('>>>> Epoch %03d (lr %.7f)', epoch, lr)

    mean_trn_metrics = np.mean(trn_metrics, axis=0)
    logging.info(
        '[Train] time: %3.2f, total_loss: %2.4f, global_loss: %2.4f, refine_loss: %2.4f',
        trn_time, trn_metrics[0], trn_metrics[1], trn_metrics[2])

    if val_time is not None:
        mean_val_metrics = np.mean(val_metrics, axis=0)
        logging.info(
            '[Valid] time: %3.2f, total_loss: %2.4f, global_loss: %2.4f, refine_loss: %2.4f',
            val_time, val_metrics[0], val_metrics[1], val_metrics[2])
