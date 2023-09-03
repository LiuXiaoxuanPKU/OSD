import pickle
import os
from dataclasses import dataclass
from typing import Any
import time

@dataclass
class Spec:
    data : Any
    timestamp: float
    name: str

class SpecLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'wb') as f:
                pass

    def log(self, name, obj):
        with open(self.log_file, 'ab') as f:
            pickle.dump(Spec(obj, time.time(), name), f)

    def read_logs(self):
        logs = []
        with open(self.log_file, 'rb') as f:
            while True:
                try:
                    logs.append(pickle.load(f))
                except EOFError:
                    break
        return logs
