import os
import yaml
import linecache
from typing import Collection, Dict

from celltk.core.operation import Operation
from celltk.utils.process_utils import condense_operations


def load_yaml(path: str, loader: str = 'Loader') -> dict:
    """
    TODO: Move all yaml loading to use this function
    """
    loader = getattr(yaml, loader, yaml.Loader)
    with open(path, 'r') as yf:
        yaml_dict = yaml.load(yf, Loader=loader)

    return yaml_dict


def save_yaml_file(data: dict, path: str, warning: bool = True) -> None:
    """
    """
    warning_text = ('This file was automatically generated. Do not edit. '
                    'To generate a new file use the methods of Pipeline '
                    'or Orchestrator.',
                    'Or use the save_operation_yaml and save_pipeline_yaml '
                    'functions in celltk.utils.file_utils.')
    with open(path, 'w') as yf:
        if warning:
            for warn in warning_text:
                yf.write('# ' + warn + '\n')
        yaml.dump(data, yf, sort_keys=False)


def save_operation_yaml(yaml_file: str,
                        operations: Collection[Operation]
                        ) -> None:
    """Save a collection of operations in a yaml file"""
    # Get Operation definitions
    op_dict = condense_operations(operations)

    # Format and save
    op_dict = {'_operations': op_dict}
    save_yaml_file(op_dict, yaml_file, warning=True)


def save_pipeline_yaml(yaml_file: str,
                       pipeline: Dict[str, str]
                       ) -> None:
    """Save a Pipeline in a yaml file"""
    save_yaml_file(pipeline, yaml_file, warning=True)


def save_job_history_yaml(yaml_file: str,
                          job_history: Dict[str, dict]
                          ) -> None:
    """Saves a record of the most recent submissions to SLURM"""
    pass


def get_file_line(path: str, lineno: str = 1) -> str:
    """
    Retrieves an arbitrary line from file.
    If negative, seeks backwards.

    NOTE: https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
    """
    if lineno >= 0:
        # NOTE: Will return an empty string on any Exception
        return linecache.getline(path, lineno).rstrip(' \n')
    else:
        lineno *= -1
        curr_line = 0
        # Seek backwards for efficiency
        with open(path, 'rb') as f:
            try:
                # Seek backwards from end until linebreak
                f.seek(-2, os.SEEK_END)
                while True:
                    f.seek(-2, os.SEEK_CUR)
                    if f.read(1) == b'\n':
                        curr_line += 1
                        if curr_line == lineno:
                            break
            except OSError:
                if curr_line < lineno:
                    # File too short, return '' as above
                    return ''
                else:
                    f.seek(0)
            return f.readline().decode().rstrip(' \n')


def folder_name(path: str) -> str:
    """Returns name of last folder in a path
    TODO: Doesn't work if path points to file - returns file name, not folder name
    """
    return os.path.basename(os.path.normpath(path))
