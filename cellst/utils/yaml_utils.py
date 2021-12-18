import yaml
from typing import Collection, Dict

from cellst.operation import Operation
# from cellst.pipeline import Pipeline, Orchestrator
from cellst.utils.process_utils import condense_operations


def save_yaml_file(data: dict, path: str, warning: bool = True) -> None:
    """
    """
    warning_text = ('This file was automatically generated. Do not edit '
                    'manually unless you are know what you are doing.',
                    'To generate a new file use the methods of Pipeline '
                    'or Orchestrator.',
                    'Or use the save_operation_yaml and save_pipeline_yaml '
                    'functions in cellst.utils.yaml_utils.')

    with open(path, 'w') as yf:
        if warning:
            for warn in warning_text:
                yf.write('# ' + warn + '\n')
        yaml.dump(data, yf)


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
