import sys
from typing import Dict, Collection

from cellst.operation import Operation

from cellst.utils._types import TYPE_LOOKUP


def condense_operations(operations: Collection[Operation]) -> Dict[str, dict]:
    """
    This function receives a list of operations and
    returns a dictionary that can recreate them
    """
    return {op.__name__.lower(): op._operation_to_dict()
            for op in operations}


def extract_operations(oper_dict: Dict) -> Operation:
    """
    The opposite function of condense operations.
    Takes in a dictionary and returns the operation object
    """
    return [_dict_to_operation(o) for o in oper_dict.values()]


def _dict_to_operation(oper_dict: Dict) -> Operation:
    """
    """
    # Get all the values that relate to the Operation
    to_init = {k: v for k, v in oper_dict.items() if k != 'FUNCTIONS'}

    # Get operation class to call
    operation = to_init.pop('__name__')
    # Seems messy. Any change to repo structure will break this.
    operation = getattr(sys.modules[f'cellst.{operation.lower()}'], operation)

    # Initalize the class
    operation = operation(**to_init)

    # Add the functions to the operation
    for key, val in oper_dict['FUNCTIONS'].items():
        func = key
        name, args, kwargs = val['name'], val['args'], val['kwargs']

        # Get the type if custom type
        exp_type = val['output_type']
        try:
            exp_type = TYPE_LOOKUP[exp_type.__name__]
        except (KeyError, AttributeError):
            pass

        try:
            # Add the functions to the operation
            operation.add_function_to_operation(func, exp_type, name,
                                                *args, **kwargs)
        except NotImplementedError:
            # Extract already has function added, but needs other info
            operation.set_metric_list(val['metrics'])
            for k, v in val['extra_props'].items():
                operation.add_extra_metric(k, v)

    return operation
