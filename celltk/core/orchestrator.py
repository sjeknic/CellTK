import os
import sys
import argparse
import yaml
import itertools
import warnings
from multiprocessing import Pool
from typing import Dict, Collection, Tuple, Callable, Union
from glob import glob

from celltk.core.operation import Operation
from celltk.core.pipeline import Pipeline
from celltk.core.arrays import ExperimentArray
from celltk.utils._types import Array
from celltk.utils.process_utils import condense_operations, extract_operations
from celltk.utils.log_utils import get_logger, get_console_logger
from celltk.utils.file_utils import (save_operation_yaml, save_pipeline_yaml,
                                     save_yaml_file, folder_name)
from celltk.utils.slurm_utils import JobController, SlurmController
from celltk.utils.cli_utils import CLIParser


class Orchestrator():
    """
    Orchestrator organizes running multiple Pipelines on a set of folders

    :param yaml_folder: Absolute path to location of Pipeline yamls
    :param parent_folder: Absolute path of the folders with raw images
    :param output_folder: Absolute path to folder to store outputs.
        Defaults to 'parent_folder/outputs'
    :param match_str: If provided, folders that do not contain
        match_str are excluded from analysis
    :param image_folder: Absolute path to folder with images
        if different from parent_folder or output_folder
    :param mask_folder: Absolute path to folder with masks
        if different from parent_folder or output_folder
    :param array_folder: Absolute path to folder with arrays
        if different from parent_folder or output_folder
    :param condition_map: Dictionoary that maps folder names too
        the condition in the experiment. i.e {A1-Site_0: control}
    :param position_map: If multiple positions have the same condition
        position map is used to uniquely identify them
    :param cond_map_only: If True, folders not in cond_map are not run.
        Only used if condition_map is provided.
    :param name: Used to identify output array.
        Defaults to name parent_folder.
    :param frame_rng: Specify the frames to be loaded and used.
        If a single int, will load that many images. Otherwise
        designates the bounds of the range.
    :param skip_frames: Use to specify frames to be skipped.
        For example, frames that are out of focus.
    :param file_extension: File extension of image files to be loaded
    :param ovewrite: If True, outputs in output_folder are overwritten
    :param log_file: If True, save log outputs to a
        text file in output folder
    :param save_master_df: If True, saves a single hdf5 file containing
        the data from all of the pipelines
    :param job_controller: If given, controls how pipelines are run
    :param verbose: If True, increases logging verbosity

    :return: None

    TODO:
        - Add __str__ method
    """
    file_location = os.path.dirname(os.path.realpath(__file__))

    __name__ = 'Orchestrator'
    __slots__ = ('pipelines', 'operations', 'yaml_folder',
                 'parent_folder', 'output_folder',
                 'operation_index', 'file_extension', 'name',
                 'overwrite', 'save', 'condition_map', 'cond_map_only',
                 'logger', 'controller', 'log_file', 'verbose',
                 'position_map', 'skip_frames', '__dict__')

    def __init__(self,
                 yaml_folder: str = None,
                 parent_folder: str = None,
                 output_folder: str = None,
                 match_str: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 array_folder: str = None,
                 condition_map: dict = {},
                 position_map: Union[dict, Callable] = None,
                 cond_map_only: bool = False,
                 name: str = 'experiment',
                 frame_rng: Tuple[int] = None,
                 skip_frames: Tuple[int] = None,
                 file_extension: str = 'tif',
                 overwrite: bool = True,
                 log_file: bool = True,
                 save_master_df: bool = True,
                 job_controller: JobController = None,
                 verbose: bool = True,
                 ) -> None:
        """
        Args:

        Returns:
            None

        TODO:
            - Should be able to parse args and load a yaml as well
            - Should be able to load yaml to define operations
        """
        # Save some values
        self.name = name
        self.frame_rng = frame_rng
        self.skip_frames = skip_frames
        self.file_extension = file_extension
        self.overwrite = overwrite
        self.save = save_master_df
        self.condition_map = condition_map
        self.position_map = self._set_position_map(position_map)
        self.cond_map_only = cond_map_only
        self.controller = job_controller
        self.log_file = log_file
        self.verbose = verbose

        # Set paths and start logging
        self._set_all_paths(yaml_folder, parent_folder, output_folder)
        self._make_output_folder(self.overwrite)
        if log_file:
            lev = 'info' if verbose else 'warning'
            self.logger = get_logger(self.__name__, self.output_folder,
                                     overwrite=overwrite, console_level=lev)
        else:
            self.logger = get_console_logger()

        # Build the Pipelines and input/output paths
        self.pipelines = {}
        self._build_pipelines(match_str, image_folder, mask_folder,
                              array_folder)

        # Prepare for getting operations
        self.operations = []

    def __len__(self) -> int:
        return len(self.pipelines)

    def run(self, n_cores: int = 1) -> None:
        """
        Load images and run all of the pipelines

        :param n_cores: Not currently implemented

        :retrun: None
        :rtype: None
        """
        # Can skip doing anything if no operations have been added
        if len(self.operations) == 0 or len(self.pipelines) == 0:
            warnings.warn('No Operations and/or Pipelines. Returning None.',
                          UserWarning)
            return

        # Add operations to the pipelines
        self._add_operations_to_pipelines(self.operations)

        # Run with multiple cores or just a single core
        if self.controller is not None:
            self.logger.info(f'Running Pipelines with {self.controller}')
            self.controller.set_logger(self.logger)
            with self.controller:
                self.controller.run(self.pipelines)
                results = []
        elif n_cores > 1:
            raise NotImplementedError('This feature still requires '
                                      'some machine-specific debugging')
            results = self._run_multiple_pipelines(self.pipelines,
                                                   n_cores=n_cores)
        else:
            results = []
            for fol, kwargs in self.pipelines.items():
                self.logger.info(f'Starting Pipeline {fol}')
                results.append(Pipeline._run_single_pipe(kwargs))

        if self.save:
            # TODO: If results are saved, pass here, otherwise, use files
            self.build_experiment_file()

        return results

    def add_operations(self,
                       operation: Collection[Operation],
                       index: int = -1
                       ) -> None:
        """
        Adds Operations to the Orchestrator and all pipelines.

        :param operation: A single operation or collection of operations
            to be run in the order passed
        :param index: If given, dictates where to insert the new operations

        :return: None
        """
        if isinstance(operation, Collection):
            if all([isinstance(o, Operation) for o in operation]):
                if index == -1:
                    self.operations.extend(operation)
                else:
                    self.operations[index:index] = operation
            else:
                raise ValueError('Not all items in operation are Operations.')
        elif isinstance(operation, Operation):
            if index == -1:
                self.operations.append(operation)
            else:
                self.operations.insert(index, operation)
        else:
            raise ValueError(f'Expected Operation, got {type(operation)}.')

        self.operation_index = {i: o for i, o in enumerate(self.operations)}

    def load_operations_from_yaml(self, path: str) -> None:
        """
        Loads Operations from a YAML file

        :param path: path to YAML file

        :return: None
        """
        with open(path, 'r') as yf:
            op_dict = yaml.load(yf, Loader=yaml.Loader)

        try:
            op_dict = op_dict['_operations']
            opers = extract_operations(op_dict)
            self.add_operations(opers)
        except KeyError:
            raise KeyError(f'Failed to find Operations in {path}.')

    def build_experiment_file(self,
                              match_str: str = None,
                              ) -> None:
        """
        Builds a single ExperimentArray from all of the ConditionArrays
        found in the Pipeline folders.

        :param match_str: If given, only files containing match_str
            are included

        TODO:
            - Increase efficiency by grouping sites by condition before loading
        """
        # Make ExperimentArray to hold data
        if match_str:
            out = ExperimentArray(name=match_str)
            name = match_str
        else:
            out = ExperimentArray(name=self.name)
            name = self.name

        self.logger.info(f'Building ExperimentArray {out}')

        # Search for all dfs in all pipeline folders
        for fol in self.pipelines:
            otpt_fol = os.path.join(self.output_folder, fol)

            # NOTE: if df.name is already in Experiment, will be overwritten
            for df in glob(os.path.join(otpt_fol, '*.hdf5')):
                if match_str and match_str not in df: continue
                out.load_condition(df)
                self.logger.info(f'Loaded {df}')

        # Merge conditions - does nothing if all are unique
        out.merge_conditions()

        # Save the master df file
        save_path = os.path.join(self.output_folder, f'{name}.hdf5')
        out.save(save_path)
        self.logger.info(f'Saved ExperimentArray at {save_path}')

    def update_condition_map(self,
                             condition_map: dict = {},
                             path: str = None,
                             ) -> None:
        """
        Adds conditions to each of the Pipelines in Orchestrator

        :param condition_map: Dictionoary that maps folder names too
            the condition in the experiment. i.e {A1-Site_0: control}
        :param path: If given, loads a condition map from the file
            found at path. Overwrites condition_map.

        :return:  None
        """
        # Load condition map YAML if available
        if path:
            _cond_map = self._load_cond_map_from_yaml(path)
            condition_map.update(_cond_map)

        for fol, cond in condition_map.items():
            self.pipelines[fol]['name'] = cond

        self.condition_map = condition_map

    def save_pipelines_as_yamls(self, path: str = None) -> None:
        """
        Saves Orchestrator configuration as a YAML file

        :param path: Path to save yaml file in. Defaults to output_folder

        :return: None
        """
        # Set path for saving files - saves in yaml folder
        if path is None:
            path = os.path.join(self.output_folder, 'pipeline_yamls')
        if not os.path.exists(path):
            os.makedirs(path)

        # Make sure operations are added to pipelines
        self._add_operations_to_pipelines(self.operations)

        # Save each pipeline
        self.logger.info(f"Saving {len(self.pipelines)} yamls in {path}")
        for pipe, kwargs in self.pipelines.items():
            # Save the Pipeline
            fname = os.path.join(path,
                                 f"{folder_name(kwargs['parent_folder'])}.yaml")
            save_pipeline_yaml(fname, kwargs)

    def save_operations_as_yaml(self,
                                path: str = None,
                                fname: str = 'operations.yaml'
                                ) -> None:
        """
        Save configuration of Operations in Pipeline as a YAML file

        :param path: Path to save yaml file in. Defaults to output_folder
        :param fname: Name of the YAML file. Defaults to operations.yaml

        :return: None
        """
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Save using file_utils
        self.logger.info(f"Saving Operations at {path}")
        save_operation_yaml(path, self.operations)

    def save_condition_map_as_yaml(self,
                                   path: str = None,
                                   fname: str = 'conditions.yaml'
                                   ) -> None:
        """
        Saves the conditions in Orchestrator as a YAML file

        :param path: Path to save the file at

        :return:  None
        """
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Save using file_utils
        self.logger.info(f"Saving condition_map at {path}")
        save_yaml_file(self.condition_map, path, warning=False)

    def _add_operations_to_pipelines(self,
                                     operations: Collection[Operation] = []
                                     ) -> None:
        """
        Adds Operations to each Pipeline in Orchestrator
        """
        # Collect operations
        op_dict = condense_operations(operations)
        self.logger.info(f'Adding Operations {operations} '
                         f'to {len(self)} Pipelines.')
        for pipe, kwargs in self.pipelines.items():
            # If Extract is in operations, update the condition
            # NOTE: This dictionary must preserve order
            op = {k if 'extract' not in k else 'extract':
                  v if 'extract' not in k else {}
                  for k, v in op_dict.items()}
            # TODO: This could be done outside of the outer loop
            for opname in op_dict:
                if 'extract' in opname:
                    # Check if the name is the default and update
                    if op_dict[opname]['condition'] == 'condition':
                        condition = kwargs['name']
                    else:
                        condition = op_dict[opname]['condition']

                    # Get position if needed for multiple positions
                    try:
                        pos = self.position_map[pipe]
                    except (KeyError, AttributeError, TypeError):
                        pos = 0

                    # Make new extract dictionary - cannot edit in place
                    new_extract = {}
                    for k, v in op_dict[opname].items():
                        if k == 'condition':
                            new_extract[k] = condition
                        elif k == 'position_id':
                            new_extract[k] = pos
                        else:
                            new_extract[k] = v

                    # Copy values to new operation dictionary
                    op.update({opname: new_extract})

            # First try to append operations before overwriting
            try:
                kwargs['_operations'].update(op)
            except KeyError:
                kwargs.update({'_operations': op})

    def _run_multiple_pipelines(self,
                                pipeline_dict: Dict,
                                n_cores: int = 1
                                ) -> Collection[Array]:
        """
        pipeline_dict holds path information for building ALL of the pipelines
            - key is subfolder, val is to be passed to Pipeline.__init__

        Assumes that operations are already added to the Pipelines
        TODO: Fix
        """
        # Run asynchronously
        with Pool(n_cores) as pool:
            # Set up pool of workers
            multi = [pool.apply_async(Pipeline._run_single_pipe, args=(kwargs))
                     for kwargs in pipeline_dict.values()]

            # Run pool and return results
            return [m.get() for m in multi]

    def _set_position_map(self, position_map: (dict, Callable)) -> dict:
        """
        Save a unique identifier for each position if needed

        If position_map is Callable, should input position name (key in
        self.condition_map) and return int
        """
        # Check for duplicate conditions
        uniq_conds = tuple(itertools.groupby(self.condition_map.values()))
        need_merge = len(uniq_conds) != len(self.condition_map)

        if isinstance(position_map, dict):
            return position_map  # assume user did it correctly
        if need_merge:
            if isinstance(position_map, Callable):
                # User passed lambda function- check it returns int
                try:
                    int(position_map(next(iter(self.condition_map))))
                    id_func = position_map
                except (TypeError, ValueError, StopIteration):
                    # Warn user and go to default
                    warnings.warn('position_map is Callable but does not return '
                                  'int. Using default positions.', UserWarning)
                    id_func = None
            else:
                warnings.warn('Did not get usable position_map. '
                              'Using default positions.', UserWarning)
                id_func = None

            # Make position map with id_func or just integers
            if id_func:
                position_map = {k: id_func(k) for k in self.condition_map}
            else:
                position_map = {k: n for n, k in enumerate(self.condition_map)}

        return position_map

    def _load_cond_map_from_yaml(self, path: str) -> dict:
        with open(path, 'r') as yf:
            cond_map = yaml.load(yf, Loader=yaml.Loader)

        return cond_map

    def _build_pipelines(self,
                         match_str: str,
                         image_folder: str,
                         mask_folder: str,
                         array_folder: str
                         ) -> None:
        """
        """
        # If yamls are provided, use those to make the Pipelines
        if self.yaml_folder is not None:
            files = [os.path.join(self.yaml_folder, f)
                     for f in os.listdir(self.yaml_folder)
                     if f.endswith('.yaml')]
            self.logger.info(f'Found {len(files)} possible pipelines '
                             f'in {self.yaml_folder}')
            # Load all yamls as dictionaries
            for f in files:
                with open(f, 'r') as yf:
                    pipe = yaml.load(yf, Loader=yaml.Loader)
                    try:
                        fol = folder_name(pipe['parent_folder'])
                        self.pipelines[fol] = pipe
                    except KeyError:
                        # Indicates yaml file was not a Pipeline yaml
                        pass
        else:
            # Find all folders that will be needed for Pipelines
            self.logger.info(f'Found {len(os.listdir(self.parent_folder))} '
                             f'possible pipelines in {self.parent_folder}')
            for fol in os.listdir(self.parent_folder):
                # Check if positions have to be in condition_map
                if self.cond_map_only and fol not in self.condition_map:
                    continue
                # Check for the match_str
                elif match_str and match_str not in fol:
                    continue

                # Make sure the path is a directory
                fol_path = os.path.join(self.parent_folder, fol)
                if os.path.isdir(fol_path) and fol_path != self.output_folder:
                    # First initialize the dictionary
                    self.pipelines[fol] = {}

                    # Save parent folder
                    self.pipelines[fol]['parent_folder'] = fol_path

                    # Save output folder relative to self.output_folder
                    out_fol = os.path.join(self.output_folder, fol)
                    self.pipelines[fol]['output_folder'] = out_fol

                    # Set all of the subfolders
                    self.pipelines[fol].update(dict(
                        image_folder=self._set_rel_to_par(fol_path,
                                                          image_folder),
                        mask_folder=self._set_rel_to_par(fol_path,
                                                         mask_folder),
                        array_folder=self._set_rel_to_par(fol_path,
                                                          array_folder),
                    ))

                    # Add condition
                    try:
                        self.pipelines[fol]['name'] = self.condition_map[fol]
                    except KeyError:
                        self.pipelines[fol]['name'] = fol

                    # Add miscellaneous options
                    self.pipelines[fol]['frame_rng'] = self.frame_rng
                    self.pipelines[fol]['skip_frames'] = self.skip_frames
                    self.pipelines[fol]['file_extension'] = self.file_extension
                    self.pipelines[fol]['overwrite'] = self.overwrite
                    self.pipelines[fol]['log_file'] = self.log_file
                    self.pipelines[fol]['verbose'] = self.verbose

        self.logger.info(f'Loaded {len(self)} pipelines')

    def _set_all_paths(self,
                       yaml_folder: str,
                       parent_folder: str,
                       output_folder: str
                       ) -> None:
        # Parent folder defaults to folder where Orchestrator was called
        if parent_folder is None:
            self.parent_folder = os.path.abspath(sys.argv[0])
        else:
            self.parent_folder = parent_folder

        # Output folder defaults to folder in parent_folder
        if output_folder is None:
            self.output_folder = os.path.join(self.parent_folder, 'outputs')
        else:
            self.output_folder = output_folder

        # YAML folder can remain None
        self.yaml_folder = yaml_folder

    def _make_output_folder(self,
                            overwrite: bool = True
                            ) -> None:
        """
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            fmode = 'a'
        elif overwrite:
            fmode = 'w'
        elif not overwrite:
            fmode = 'a'
            it = 0
            tempdir = self.output_folder + f'_{it}'
            while os.path.exists(tempdir):
                it += 1
                tempdir = self.output_folder + f'_{it}'

            self.output_folder = tempdir
            os.makedirs(self.output_folder)

    def _set_rel_to_par(self,
                        par_path: str,
                        inpt: str
                        ) -> str:
        """
        Makes inpts relative to the parent_folder of that Pipeline
        """
        # Parse sub-folder inputs
        if inpt is not None:
            inpt = os.path.join(par_path, inpt)

        return inpt

    @classmethod
    def _build_from_cli(cls, args: argparse.Namespace) -> 'Orchestrator':
        """
        Work in progress. Still needs to be finished.
        """
        # Build the Orchestrator
        input_args = dict(
            yaml_folder=args.pipelines,
            parent_folder=args.parent,
            output_folder=args.output,
            match_str=args.match_str,
            image_folder=args.image,
            mask_folder=args.mask,
            array_folder=args.array,
            file_extension=args.extension,
            name=args.name,
            overwrite=args.overwrite,
            log_file=not args.no_log,
            save_master_df=not args.no_save_master_df,
        )

        # Try building a Controller
        try:
            controller_args = dict(
                partition=args.partition,
                user=args.user,
                time=args.time,
                cpu=args.cpus,
                mem=args.mem,
                name=args.job_name,
                modules=args.modules,
                maxjobs=args.maxjobs
            )

        except AttributeError:
            raise AttributeError()

        scont = SlurmController(**controller_args)
        orch = Orchestrator(**input_args, job_controller=scont)

        # Add operations
        if args.operations is not None:
            orch.load_operations_from_yaml(args.operations)

        return orch


if __name__ == '__main__':
    parser = CLIParser('orchestrator')
    args = parser.get_command_line_inputs()

    orch = Orchestrator._build_from_cli(args)
    orch.run()

