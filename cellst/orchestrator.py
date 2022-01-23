import os
import sys
import argparse
import yaml
import warnings
from multiprocessing import Pool
from typing import Dict, Collection
from glob import glob

from cellst.operation import Operation
from cellst.pipeline import Pipeline
from cellst.utils._types import Arr, Experiment
from cellst.utils.process_utils import condense_operations, extract_operations
from cellst.utils.utils import folder_name
from cellst.utils.log_utils import get_logger, get_console_logger
from cellst.utils.yaml_utils import (save_operation_yaml, save_pipeline_yaml,
                                     save_yaml_file)
from cellst.utils.slurm_utils import JobController, SlurmController
from cellst.utils.cli_utils import CLIParser


class Orchestrator():
    """
    TODO:
        - Add __str__ method
        - Add function for submitting jobs to SLURM
    """
    file_location = os.path.dirname(os.path.realpath(__file__))

    __name__ = 'Orchestrator'
    __slots__ = ('pipelines', 'operations', 'yaml_folder',
                 'parent_folder', 'output_folder',
                 'operation_index', 'file_extension', 'name',
                 'overwrite', 'save', 'condition_map',
                 'logger', 'controller')

    def __init__(self,
                 yaml_folder: str = None,
                 parent_folder: str = None,
                 output_folder: str = None,
                 match_str: str = None,
                 image_folder: str = None,
                 mask_folder: str = None,
                 track_folder: str = None,
                 array_folder: str = None,
                 condition_map: dict = {},
                 name: str = 'experiment',
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
        self.file_extension = file_extension
        self.overwrite = overwrite
        self.save = save_master_df
        self.condition_map = condition_map
        self.controller = job_controller

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
                              track_folder, array_folder)

        # Prepare for getting operations
        self.operations = []

    def __len__(self) -> int:
        return len(self.pipelines)

    def run(self, n_cores: int = 1) -> None:
        """
        Run all the Pipelines with all of the operations.

        TODO:
            - Should results really be saved?
        """
        # Can skip doing anything if no operations have been added
        if len(self.operations) == 0 or len(self.pipelines) == 0:
            warnings.warn('No Operations and/or Pipelines. Returning None.',
                          UserWarning)
            return

        # Add operations to the pipelines
        self.add_operations_to_pipelines(self.operations)

        # Run with multiple cores or just a single core
        if self.controller is not None:
            self.logger.info(f'Running Pipelines with {self.controller}')
            self.controller.set_logger(self.logger)
            with self.controller:
                self.controller.run(self.pipelines)
                results = []
        elif n_cores > 1:
            results = self.run_multiple_pipelines(self.pipelines,
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

    def run_multiple_pipelines(self,
                               pipeline_dict: Dict,
                               n_cores: int = 1
                               ) -> Collection[Arr]:
        """
        pipeline_dict holds path information for building ALL of the pipelines
            - key is subfolder, val is to be passed to Pipeline.__init__

        Assumes that operations are already added to the Pipelines
        """
        # Run asynchronously
        with Pool(n_cores) as pool:
            # Set up pool of workers
            multi = [pool.apply_async(Pipeline._run_single_pipe, args=(kwargs))
                     for kwargs in pipeline_dict.values()]

            # Run pool and return results
            return [m.get() for m in multi]

    def add_operations(self,
                       operation: Collection[Operation],
                       index: int = -1
                       ) -> None:
        """
        Adds Operations to the Orchestrator and recalculates the index

        Args:

        Returns:
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
        with open(path, 'r') as yf:
            op_dict = yaml.load(yf, Loader=yaml.Loader)

        try:
            op_dict = op_dict['_operations']
            opers = extract_operations(op_dict)
            self.add_operations(opers)
        except KeyError:
            raise KeyError(f'Failed to find Operations in {path}.')

    def build_experiment_file(self, arrays: Collection[Arr] = None) -> None:
        """
        Search folders in self.pipelines for hdf5 data frames
        """
        # Make Experiment array to hold data
        out = Experiment(name=self.name)

        # Search for all dfs in all pipeline folders
        for fol in self.pipelines:
            otpt_fol = os.path.join(self.output_folder, fol)

            # NOTE: if df.name is already in Experiment, will be overwritten
            for df in glob(os.path.join(otpt_fol, '*.hdf5')):
                out.load_condition(df)

        # Save the master df file
        save_path = os.path.join(self.output_folder, f'{self.name}.hdf5')
        out.save(save_path)

    def add_operations_to_pipelines(self,
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
            # TODO: This won't work for multiple Extract operations
            if 'extract' in op_dict and op_dict['extract']['condition'] == 'default':
                condition = kwargs['name']
                new_extract = {k: (v if k != 'condition' else pipe)
                               for k, v in op_dict['extract'].items()}
                op = {k: v for k, v in op_dict.items() if k != 'extract'}
                op.update({'extract': new_extract})
            else:
                op = op_dict

            # First try to append operations before overwriting
            try:
                kwargs['_operations'].update(op)
            except KeyError:
                kwargs.update({'_operations': op})

    def update_condition_map(self,
                             condition_map: dict = {},
                             path: str = None,
                             ) -> None:
        """
        Adds conditions to each of the Pipelines in Orchestrator

        NOTE: path will overwrite anything in condition_map
        """
        # Load condition map YAML if available
        if path is not None:
            _cond_map = self._load_cond_map_from_yaml(path)
            condition_map.update(_cond_map)

        for fol, cond in condition_map.items():
            self.pipelines[fol]['name'] = cond

        self.condition_map = condition_map

    def save_pipelines_as_yamls(self, path: str = None) -> None:
        """
        Save yaml file that can be loaded as Pipeline
        """
        # Set path for saving files - saves in yaml folder
        if path is None:
            path = os.path.join(self.output_folder, 'pipeline_yamls')
        if not os.path.exists(path):
            os.makedirs(path)

        # Make sure operations are added to pipelines
        self.add_operations_to_pipelines(self.operations)

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
        Save self.operations as a yaml file.
        """
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, fname)

        # Save using yaml_utils
        self.logger.info(f"Saving Operations at {path}")
        save_operation_yaml(path, self.operations)

    def save_condition_map_as_yaml(self, path: str = None) -> None:
        # Get the path
        path = self.output_folder if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        # Save using yaml_utils
        self.logger.info(f"Saving condition_map at {path}")
        save_yaml_file(self.condition_map, path, warning=False)

    def _load_cond_map_from_yaml(self, path: str) -> dict:
        with open(path, 'r') as yf:
            cond_map = yaml.load(yf, Loader=yaml.Loader)

        return cond_map

    def _build_pipelines(self,
                         match_str: str,
                         image_folder: str,
                         mask_folder: str,
                         track_folder: str,
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
                    pipe = yaml.load(yf, Loader=yaml.FullLoader)
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
                # Check for the match_str
                if match_str is not None and match_str not in fol:
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
                        track_folder=self._set_rel_to_par(fol_path,
                                                          track_folder),
                        array_folder=self._set_rel_to_par(fol_path,
                                                          array_folder),
                    ))

                    # Add condition
                    try:
                        self.pipelines[fol]['name'] = self.condition_map[fol]
                    except KeyError:
                        self.pipelines[fol]['name'] = fol

                    # Add miscellaneous options
                    self.pipelines[fol]['file_extension'] = self.file_extension
                    self.pipelines[fol]['overwrite'] = self.overwrite

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
            track_folder=args.track,
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

