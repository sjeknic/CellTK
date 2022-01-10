import subprocess
import logging
import os
from time import sleep
from typing import Collection

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
    _delay = 10  # Wait between commands in sec

    def __init__(self,
                 pipelines: Collection[dict],
                 partition: str,
                 user: str,
                 time: str,
                 cpu: int,
                 mem: str,
                 name: str = 'cellst',
                 modules: str = None,
                 maxjobs: int = 1,
                 ) -> None:
        # Save the inputs
        self.pipelines = pipelines
        self.partition = partition
        self.user = user
        self.time = time
        self.cpu = cpu
        self.mem = mem
        self.name = name
        self.modules = modules
        self.maxjobs = maxjobs

        # Set up default logger
        self.logger = get_console_logger()

    def __enter__(self):
        print(os.getcwd())
        self._working_dir = os.path.join(os.getcwd(), '.cellst_temp')
        if not os.path.exists(self._working_dir):
            os.makedirs(self._working_dir)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if os.path.exists(self._working_dir):
            # This might raise an error if dir is not empty
            os.rmdir(self._working_dir)
        del self._working_dir

    def run(self, pipelines: dict) -> None:
        """This needs to run the individual pipelines"""
        # TODO: How to pass the pipeline yaml without making the file???
        #       or should I just accept it and always make the yaml in the output dir?
        curr_jobs = self._check_user_jobs(self.partition, self.user)
        print('current jobs ', curr_jobs)

        if curr_jobs < self.maxjobs:
            sbatch_path = self._yield_working_sbatch(pipelines)

            # Run the sbatch
            subprocess.run(f"sbatch {sbatch_path}", shell=True)

            # Wait
            print('submitted and waiting')
            sleep(self._delay)

    def _yield_working_sbatch(self, pipelines: dict):
        for fol, kwargs in pipelines.items():
            print(fol)
            # First load the pipeline, then save as yaml that can be accessed
            # TODO: Obviously could be more efficient - esp if Orchestrator made yamls
            _pipe = Pipeline._build_from_dict(kwargs)
            _y_path = os.path.join(self._working_dir, 'tempyaml.yaml')
            _pipe.save_as_yaml(self._working_dir, 'tempyaml.yaml')

            # Make batch script
            _batch_path = os.path.join(self._working_dir, 'tempsbatch.sh')
            self._create_bash_script(partition=self.partition, job_name=f'{self.name}_{fol}',
                                     time=self.time, ntasks=1, cpus_per_task=self.cpu,
                                     mem=self.mem, fname=_batch_path, yaml_path=_y_path)

            # Return path to the script
            yield _batch_path

            # After running, clean up the files
            os.remove(_y_path, _batch_path)

    def _check_user_jobs(self,
                         partition: str = '$GROUP',
                         user: str = '$USER'
                         ) -> int:
        """
        Checks the SLURM queue and returns the number of jobs
        """
        # Count jobs using the number of lines - does not distinguish running or not
        command = f'squeue -p {partition} -u {user} | wc -l'
        result = subprocess.run([command], shell=True)  # what to do with stdout?

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
                            ) -> None:
        """
        Runs a bash script to submit a single job to the SLURM controller
        sbatch will return 0 on success or error code on failure.
        """
        def _add_line(add: str,
                      header: str = '#SBATCH',
                      div: str = ' '
                      ) -> str:
            string += f'\n{header}{div}{add}'

        # Get the options for the script
        params = {l: v for l, v in locals().items()
                  if l != 'self' and l != '_add_line'}

        # Start with bash
        string = '#!/bin/bash'
        for p, val in params.items():
            pname = p.replace('_', '-')
            to_add = f'--{pname}={val}'
            _add_line(to_add, div=' ')

        # TODO: Is the blank line really needed?
        _add_line('', '', '')

        # TODO: Add the real python command here
        mods = 'module load cellst376'
        _add_line(mods, '', '')

        command = f'python -m cellst.pipeline -y {yaml_path}'
        _add_line(command, '', '')

        # Make the file
        with open(fname, 'w') as file:
            file.write(string)

