import os
import pandas as pd
import sys
from contextlib import contextmanager
from torch.utils.data import Dataset


def relative_path(filepath):
    """Get relative path with respect to whatever file the calling code is in."""
    caller__file__ = sys._getframe(1).f_globals["__file__"]
    caller_dirname = os.path.dirname(caller__file__)
    return os.path.join(caller_dirname, filepath)


def validate_required_params(params, accept_blank_string=False):
    """Validates a dict of required params and throws the first error found if any.

    Args:
        params: Dict containing required params in the form {'param_name': param_name, ...}
    """

    for name, param in params.items():
        if param is None or accept_blank_string or param == "":
            raise ValueError(f"Missing required parameter '{name}'.")


def dict_subset(original, subset_keys):
    subset = {}
    for key, value in original.items():
        if key in subset_keys:
            subset[key] = value
    return subset


def base_object():
    return type("basic_object", (object,), {})()


def write_csv_line(file_object, items):
    """
    Given an open file object and list of strings, add them as a properly-formatted and
    escaped line to the CSV file object.

    @param file_object (io.TextIOWrapper): a target file object to write the line of CSV data
    @param items (List[str]): a list of strings to be added in order as a line to the CSV file
    @returns csv_line: the actual csv_line (including newline) that was written to the CSV file
    """

    # Format the line to write
    csv_line = (
        # Escape any double quotes as double double quotes
        ",".join(['"' + value.replace('"', '""') + '"' for value in items])
        + "\n"
    )

    file_object.write(csv_line)

    # Return formatted line back to caller
    return csv_line


class CSVSingleColumnDataset(Dataset):
    def __init__(self, path, col_name):
        """Load a single column from a CSV file into a Pandas DataFrame

        Args:
            path:
                Path to the CSV file.
            col_name:
                Column name to load. CSV file first line must contain column names.
        """
        self._df = pd.read_csv(path, dtype=str, header="infer", usecols=[col_name])
        self._col_name = col_name

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index):
        return self._df.iloc[index][self._col_name]
