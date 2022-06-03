import sys
from typing import Dict, Collection

from celltk.core.operation import Operation
from celltk.utils._types import TYPE_LOOKUP


def condense_operations(operations: Collection[Operation]) -> Dict[str, dict]:
    """
    This function receives a list of operations and
    returns a dictionary that can recreate them
    """

    # Build the dictionary iteratively
    op_dict = {}
    for op in operations:
        # Count used to identify the operations
        count = 1
        key = op.__name__.lower()
        while key in op_dict:
            key = op.__name__.lower() + f'_{count}'
            count += 1

        op_dict[key] = op._operation_to_dict()

    return op_dict


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
    to_init = {k: v for k, v in oper_dict.items() if k != '_functions'}

    # Get operation class to call
    operation = to_init.pop('__name__')
    module = to_init.pop('__module__')
    if module not in sys.modules: __import__(module)
    operation = getattr(sys.modules[module], operation)

    # Initalize the class
    operation = operation(**to_init)

    # Add the functions to the operation
    for key, val in oper_dict['_functions'].items():
        func, name, kwargs = val['func'], val['name'], val['kwargs']

        # Get the type if custom type
        exp_type = val['output_type']
        try:
            exp_type = TYPE_LOOKUP[exp_type.__name__]
        except (KeyError, AttributeError):
            pass

        try:
            # Add the functions to the operation
            operation.add_function(func, save_as=name,
                                   output_type=exp_type, **kwargs)
        except NotImplementedError:  # Means operation is Extract
            # TODO: These are all very messy and liable to break. Make neater.
            # Save other user defined parameters
            operation.set_metric_list(val['metrics'])
            # (func, keys, inverse, propagate, fr, args, kwargs)
            for nm, (f, ky, i, p, fr, a, k) in val['derived_metrics'].items():
                operation.add_derived_metric(nm, ky, f, i, p, fr, *a, **k)

            for filt in val['filters']:
                # Annoying, but otherwise args are hard to unpack
                operation.add_filter(filter_name=filt['filter_name'],
                                     metric=filt['metric'],
                                     region=filt['region'],
                                     channel=filt['channel'],
                                     frame_rng=filt['frame_rng'],
                                     *filt['args'], **filt['kwargs'])

            for k, v in val['extra_props'].items():
                operation.add_extra_metric(k, v)

    return operation
