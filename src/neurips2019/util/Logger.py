from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Queue
import os
import json
import time
import numpy as np
from enum import Enum


class LogType(Enum):
    """For better readbilty defines all types of logging"""
    SCALAR=1
    HISTOGRAM=2
    IMAGE=3

"""Streamline entries put into the logging queue"""
LogEntry = namedtuple('LogEntry', ['type', 'key', 'value', 'step', 'kwargs'])

class Logger:
    '''
    Class that manages logging statistics and makes them accessible to Tensorboard, utility for weight logging, saving agents.

    Implements a run method which should be targeted by a thread / process. Evaluates the multiprocessing queue until None is encountered.
    '''

    def __init__(self, directory:str, queue:Queue):
        """
        Constructs a logger that can accept input from several sources and log it into a tensorboard writer

        Given a directory where to store the logs and a multiprocessing queue, constructs a tensorboard log wrapper. It takes `LogEntry`s from the queue, the type of tensorboard enrty to generate is encoded in the `type` parameter which should be one of the parameters of the enum `LogType`. The rest of `LogEntry` is passed to the tensorboard function call.

        Args:
            directory: a string of the path where to log to
            queue: a multiprocessing Queue to read from
            config: a configuration file for the agent that is being logged
        """
        path = os.path.abspath(directory) # in case working directory changes elsewhere keep logging location
        if not os.path.isdir(path):
            os.makedirs(path)
        self.tb_writer = SummaryWriter(path)
        self.directory = path
        self.q = queue

    def run(self):
        """
        Method to be started by a Thread / Process. Keeps checking the logging queue until None is encountered.

        Always shut logger down by sending None, so writer is properly closed
        """
        # maps LogType entries to tensorboard functions
        type_switch = {
            LogType.SCALAR : self.tb_writer.add_scalar,
            LogType.HISTOGRAM : self.tb_writer.add_histogram,
            LogType.IMAGE : self.tb_writer.add_image
        }

        while True:
            entry = self.q.get()
            if entry is None: # shutdown signal
                self.tb_writer.close()
                break
            # log entry
            func = type_switch[entry.type]
            func(entry.key, entry.value, global_step=entry.step, **entry.kwargs)
