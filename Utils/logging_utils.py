import os
import logging
from io import StringIO
from datetime import datetime
import base64
from pathlib import Path


_PATH_TO_LOGS = os.path.join(Path(__file__).resolve().parent.parent, "logs_and_notes", "logs")

if not os.path.exists(_PATH_TO_LOGS):
    os.mkdir(_PATH_TO_LOGS)

def _get_short_datetime():
    return datetime.now().strftime("%Y%m%d_%H")

def _short_alphabetic_hash(s):
    hash_bytes = s.encode("utf-8")
    return base64.urlsafe_b64encode(hash_bytes).decode("utf-8").rstrip("=")[:10]

class MyLoggerForFailures:
    def __init__(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Use a memory handler to buffer logs
        self.log_buffer = StringIO()
        stream_handler = logging.StreamHandler(self.log_buffer)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        self.logger = logger

    def flush_log_to_file(self, filename=None):
        if filename is None:
            filename = _short_alphabetic_hash(self.logger.name)
        log_content = self.log_buffer.getvalue()
        if log_content:  # Only write if there's something logged
            with open(os.path.join(_PATH_TO_LOGS, f"{filename}_{_get_short_datetime()}.txt"), "a") as f:
                f.write(log_content)

    def log(self, to_log):
        self.logger.info(to_log)
