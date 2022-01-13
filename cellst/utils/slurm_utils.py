import subprocess
import logging
import os
from time import sleep
from typing import Collection, Generator

from cellst.utils.log_utils import get_console_logger
from cellst.pipeline import Pipeline


class JobController():
    __name__ = 'JobController'

    def set_logger(self, logger: logging.Logger) -> None:
        """
        """
        # logger is either a Pipeline or Operation logger
        log_name = logger.name

        # This logs to same file, but records the Operation name
        self.logger = logging.getLogger(f'{log_name}.{self.__name__}')


class SlurmController(JobController):
    """
    TODO:
        - This should be able to run to multiple partitions at once. SEe this ntoe.
            '''A couple things to keep in mind with running Orchestrator. Lets say that I want
    to submit jobs to both mcovert and owners. Then I need a way to 1) not submit the same job twice
    and 2) not write to the .temp.sh file from two places at once. The way I Ithink this would work best
    is if when the SlurmController is getting ready to submit a job, it first moves the yaml file to the target
    directory, makes the .temp.sh file in the target directory, then submits the sbatch request. If the job fails,
    which is quite possible. I need a way to monitor that ideally. '''

    Shitty gurobi license error:
    'gurobipy.GurobiError: HostID mismatch (licensed to cab1ca21, hostid is 59bdde52)'
    """
    __name__ = 'SlurmController'
    _line_offset = 1  # Number of lines squeue returns if n_jobs == 0
    _delay = 5  # Wait between commands in sec

    def __init__(self,
                 partition: str = '$GROUP',
                 user: str = '$USER',
                 time: str = '24:00:00',
                 cpu: int = 2,
                 mem: str = '8GB',
                 name: str = 'cst',
                 modules: str = None,
                 maxjobs: int = 1,
                 working_dir: str = '.cellst_temp',
                 output_dir: str = 'slurm_logs'
                 ) -> None:
        # Save the inputs
        self.partition = partition
        self.user = user
        self.time = time
        self.cpu = cpu
        self.mem = mem
        self.name = name
        self.modules = modules
        self.maxjobs = maxjobs

        # Save folders
        self.working_dir = os.path.join(os.getcwd(), working_dir)
        self.output_dir = os.path.join(os.getcwd(), output_dir)

        # Set up default logger
        self.logger = get_console_logger()

    def __enter__(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f'Made temporary working directory: {self.working_dir}')
        else:
            self.logger.info(f'Using existing working directory: {self.working_dir}')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f'Made output directory: {self.output_dir}')
        else:
            self.logger.info(f'Using existing output directory: {self.output_dir}')

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # if os.path.exists(self.working_dir):
        #     # This might raise an error if dir is not empty
        #     os.rmdir(self.working_dir)
        # del self.working_dir
        pass

    def run(self, pipelines: dict) -> None:
        """This needs to run the individual pipelines"""
        # TODO: How to pass the pipeline yaml without making the file???
        #       or should I just accept it and always make the yaml in the output dir?
        batches = self._yield_working_sbatch(pipelines)

        _running = True
        while _running:
            curr_jobs = self._check_user_jobs(self.partition, self.user)
            self.logger.info(f'Found {curr_jobs} jobs in {self.partition}. '
                             f'Max jobs allowed: {self.maxjobs}')

            _running = self._start_single_jobs(curr_jobs, batches)

            self.logger.info(f'Finished round of submissions. Waiting {10 * self._delay}s.')
            sleep(10 * self._delay)

        self.logger.info('Finished running jobs.')

    def _start_single_jobs(self, curr_jobs: int, batches: Generator) -> None:
        """Interacts with the SLURM scheduler"""
        while curr_jobs < self.maxjobs:
            try:
                # Get the sbatch.sh file
                sbatch_path = next(batches)
            except StopIteration:
                return False

            # Run the sbatch
            subprocess.run([f"sbatch {sbatch_path}"], shell=True)
            self.logger.info(f'Submitted {sbatch_path}. Waiting {self._delay}s.')
            sleep(self._delay)

            curr_jobs += 1

        return True

    def _yield_working_sbatch(self, pipelines: dict):
        self.logger.info(f'Building {len(pipelines)} pipelines: {list(pipelines.keys())}')

        for pidx, (fol, kwargs) in enumerate(pipelines.items()):
            # First load the pipeline, then save as yaml that can be accessed
            # TODO: Obviously could be more efficient - esp if Orchestrator made yamls
            _pipe = Pipeline._build_from_dict(kwargs)

            _y_path = os.path.join(self.working_dir, f'{fol}yaml.yaml')
            _pipe.save_as_yaml(self.working_dir, f'{fol}yaml.yaml')

            # Make batch script
            _batch_path = os.path.join(self.working_dir, f'{fol}sbatch.sh')
            self._create_bash_script(partition=self.partition, job_name=f'{self.name}_{fol}',
                                     time=self.time, ntasks=1, cpus_per_task=self.cpu,
                                     mem=self.mem, fname=_batch_path, yaml_path=_y_path,
                                     output_path=self.output_dir)

            # Return path to the script
            yield _batch_path

    def _check_user_jobs(self,
                         partition: str = '$GROUP',
                         user: str = '$USER'
                         ) -> int:
        """
        Checks the SLURM queue and returns the number of jobs
        """
        # Count jobs using the number of lines - does not distinguish running or not
        command = f'squeue -p {partition} -u {user} | wc -l'
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE)  # what to do with stdout?

        return int(result.stdout) - self._line_offset

    def _create_bash_script(self,
                            partition: str = '$GROUP',
                            job_name: str = 'cellst_pipe',
                            time: str = '24:00:00',
                            ntasks: str = '1',
                            cpus_per_task: str = '4',
                            mem: str = '12GB',
                            fname: str = '.temp.sh',
                            yaml_path: str = None,
                            output_path: str = None
                            ) -> None:
        """
        Runs a bash script to submit a single job to the SLURM controller
        sbatch will return 0 on success or error code on failure.
        """
        def _add_line(string: str,
                      add: str,
                      header: str = '#SBATCH',
                      div: str = ' '
                      ) -> str:
            string += f'\n{header}{div}{add}'
            return string

        # Get the options for the script
        to_write = ['partition', 'job_name', 'time', 'ntasks', 'cpus_per_task',
                    'mem']
        loc = locals()
        params = {l: loc[l] for l in to_write}

        if params['partition'][0] == '$':
            params['partition'] = os.environ[params['partition'][1:]]

        # Start with bash
        string = '#!/bin/bash'
        for p, val in params.items():
            pname = p.replace('_', '-')
            to_add = f'--{pname}={val}'
            string = _add_line(string, to_add, div=' ')

        # Add output locations
        if output_path is not None:
            out = os.path.join(output_path, f'{job_name}-"%j".out')
            err = os.path.join(output_path, f'{job_name}-"%j".err')
            string = _add_line(string, f'--output={out}', div=' ')
            string = _add_line(string, f'--error={err}', div=' ')

        # TODO: Is the blank line really needed?
        string = _add_line(string, '', '', '')

        if self.modules:
            mods = f'module restore {self.modules}'
            string = _add_line(string, mods, '', '')

        command = f'python3 -m cellst.pipeline -y {yaml_path}'
        string = _add_line(string, command, '', '')

        # Make the file
        with open(fname, 'w') as file:
            file.write(string)

        self.logger.info(f'Created sbatch script at {fname}')

