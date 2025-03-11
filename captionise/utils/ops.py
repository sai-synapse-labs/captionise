import os
import sys
import signal
import random
import shutil
import requests
import functools
import traceback
import subprocess
import pickle as pkl
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from captionise.protocol import JobSubmissionSynapse   # optional if you have a specialized Synapse
from captionise.utils.logger import logger



class TimeoutException(Exception):
    """Raised when a function exceeds the given time limit."""
    pass


class ValidationError(Exception):
    """Generic validation error for data or arguments."""
    def __init__(self, message="Validation error occurred"):
        super().__init__(message)


class RsyncException(Exception):
    """Placeholder for any exceptions related to remote sync or external transfers."""
    def __init__(self, message="Rsync error occurred"):
        super().__init__(message)


def timeout_handler(seconds, func_name):
    """Handler to raise TimeoutException once the alarm signal is received."""
    raise TimeoutException(f"Function '{func_name}' timed out after {seconds} seconds.")


def timeout(seconds):
    """
    Decorator to time-limit a function. If it runs longer than 'seconds', a TimeoutException is raised.
    NOTE: Works only on Unix-like systems that support signals.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, lambda signum, frame: timeout_handler(seconds, func.__name__))
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result
        return wrapper
    return decorator


def print_on_retry(retry_state):
    """
    A helper function for tenacity-based retries, logging the attempt number.
    """
    function_name = retry_state.fn.__name__
    max_retries = retry_state.retry_object.stop.max_attempt_number
    logger.warning(
        f"Retrying {function_name}: attempt #{retry_state.attempt_number} out of {max_retries}"
    )


def delete_directory(directory: str):
    """Remove an entire directory tree."""
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        logger.info(f"Deleted directory: {directory}")


def write_pkl(data, path: str, mode="wb"):
    """Write Python object 'data' to a pickle file at 'path'."""
    with open(path, mode) as f:
        pkl.dump(data, f)


def load_pkl(path: str, mode="rb"):
    """Load and return a Python object from a pickle file at 'path'."""
    with open(path, mode) as f:
        return pkl.load(f)


def check_if_directory_exists(output_directory: str):
    """Ensure that 'output_directory' exists; if not, create it."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        logger.debug(f"Created directory {output_directory!r}")


def get_tracebacks():
    """Return and log the current exception traceback details."""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)

    logger.error(" ---------------- Traceback details ---------------- ")
    logger.warning("".join(formatted_traceback))
    logger.warning(" ---------------- End of Traceback ----------------\n")


def get_response_info(responses: List[JobSubmissionSynapse]) -> Dict:
    """Gather all desired response information from the set of miners."""
    """
    Example aggregator for response objects from your miners.
    Adjust fields as needed to gather e.g. times, status codes, or output data.
    """
    response_times = []
    status_messages = []
    status_codes = []
    returned_keys = []
    returned_sizes = []

    for resp in responses:
        # Example usage if 'resp' has e.g. resp.dendrite.process_time, etc.
        # These fields are placeholders; adjust to your actual data structure.
        response_times.append(getattr(resp, "process_time", 0))
        status_messages.append(str(getattr(resp, "status_message", "N/A")))
        status_codes.append(str(getattr(resp, "status_code", "N/A")))

        # If you store output as e.g. a dictionary in 'resp.output'
        output_data = getattr(resp, "output", {})
        returned_keys.append(list(output_data.keys()))
        returned_sizes.append([len(val) for val in output_data.values()])

    return {
        "response_times": response_times,
        "status_messages": status_messages,
        "status_codes": status_codes,
        "returned_keys": returned_keys,
        "returned_sizes": returned_sizes,
    }
