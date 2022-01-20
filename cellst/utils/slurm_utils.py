import subprocess
import logging
import os
import time as time_module
from time import sleep
from typing import Collection, Generator, List
from pprint import pp

from cellst.utils.log_utils import get_console_logger
from cellst.utils.yaml_utils import save_job_history_yaml, get_file_line, load_yaml
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
        - Add ability to set maximum number of submissions
        - Add ability to re-submit failed jobs
        - Add output of summary statistics regularly.
    """
    __name__ = 'SlurmController'
    _submit_delay = 5  # Wait between commands in sec
    _update_delay = 15  # Wait time between allowing squeue commands

    def __init__(self,
                 partition: (list, str) = '$GROUP',
                 user: str = '$USER',
                 time: str = '24:00:00',
                 cpu: int = 2,
                 mem: str = '8GB',
                 name: str = 'cst',
                 modules: str = None,
                 maxjobs: (list, int) = 1,
                 working_dir: str = '.cellst_temp',
                 output_dir: str = 'slurm_logs'
                 ) -> None:
        # Save inputs
        self.user = user
        self.name = name
        self.modules = modules
        self.maxjobs = maxjobs
        self.working_dir = os.path.join(os.getcwd(), working_dir)
        self.output_dir = os.path.join(os.getcwd(), output_dir)

        # Save the inputs
        if isinstance(partition, str):
            partition = [partition]
        self.slurm_partitions = self._make_slurm_kwargs(partition, time, cpu, mem)

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

        # Initialize some params that will be needed
        self.pipes_run = 0
        self.job_history = {}
        self.last_update_time = 0

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def run(self, pipelines: dict) -> None:
        """This needs to run the individual pipelines"""
        # TODO: How to pass the pipeline yaml without making the file???
        #       or should I just accept it and always make the yaml in the output dir?
        self.total_pipes = len(pipelines)
        batches = self._yield_working_sbatch(pipelines)

        _running = True
        while _running:
            for pidx, partition in enumerate(self.slurm_partitions):
                part_name = partition['partition']
                curr_jobs = self._check_user_jobs(part_name, self.user)
                self.logger.info(f'Found {curr_jobs}. Max '
                                 f'jobs allowed: {self.maxjobs[pidx]}')

                _running = self._submit_jobs_to_slurm(curr_jobs, self.maxjobs[pidx],
                                                      batches, partition)

            self.logger.info('Finished round of submissions. '
                             f'Sleeping {10 * self._submit_delay}s.')
            self.logger.info(f'Submitted {self.pipes_run + 1} out '
                             f'of {self.total_pipes} pipelines.')
            sleep(10 * self._submit_delay)

        self.logger.info('Finished running jobs.')

    def _make_slurm_kwargs(self,
                           partition: List,
                           time: str,
                           cpu: int,
                           mem: str,
                           ) -> List[dict]:
        """
        Makes kwargs to submit jobs to multiple partitions in the same loop

        For now, only partition can be changed (i.e. cpu, time, etc. is the same)

        TODO:
            - Allow partition specific job params
        """
        partition_list = []
        for part in partition:
            partition_list.append(dict(partition=part,
                                       time=time,
                                       cpus_per_task=cpu,
                                       mem=mem,
                                       ntasks=1,
                                       output_path=self.output_dir))

        # Need maxjobs for each partition
        if isinstance(self.maxjobs, (int, float)):
            self.maxjobs = [self.maxjobs] * len(partition)
        else:
            assert len(self.maxjobs) == len(partition_list)

        return partition_list

    def _submit_jobs_to_slurm(self,
                              curr_jobs: int,
                              max_jobs: int,
                              batches: Generator,
                              slurm_kwargs: dict
                              ) -> bool:
        """Interacts with the SLURM scheduler"""
        while curr_jobs < max_jobs:
            try:
                # Send info to batches and get sbatch file
                next(batches)
                sbatch_path = batches.send(slurm_kwargs)
            except StopIteration:
                return False

            # Run the sbatch
            submitted = subprocess.run([f"sbatch {sbatch_path}"], shell=True,
                                        capture_output=True, text=True)
            self._record_job_history(submitted)

            self.logger.info(f'Submitted {sbatch_path}. '
                             f'Sleeping {self._submit_delay}s.')
            sleep(self._submit_delay)

            curr_jobs += 1

        return True

    def _yield_working_sbatch(self, pipelines: dict):
        self.logger.info(f'Building {len(pipelines)} pipelines: {list(pipelines.keys())}')

        for self.pipes_run, (fol, kwargs) in enumerate(pipelines.items()):
            # First load the pipeline, then save as yaml that can be accessed
            # TODO: Obviously could be more efficient - esp if Orchestrator made yamls
            _pipe = Pipeline._build_from_dict(kwargs)

            _y_path = os.path.join(self.working_dir, f'{fol}yaml.yaml')
            _pipe.save_as_yaml(self.working_dir, f'{fol}yaml.yaml')

            # Make batch script
            _batch_path = os.path.join(self.working_dir, f'{fol}sbatch.sh')
            slurm_kwargs = yield
            job_name = f"{fol}_{self.name}"
            self._create_bash_script(**slurm_kwargs, fname=_batch_path,
                                     yaml_path=_y_path, job_name=job_name)

            # Return path to the script
            yield _batch_path

    def _record_job_history(self, submitted: str = None, update: bool = False) -> None:
        """
        Will record a new job (submitted) or update current record of jobs (update)
        """
        # Recorded keys
        keys = ['jobid', 'state', 'exitcode', 'runtime']

        if submitted:
            # Parse the input yaml to get values
            pth = submitted.args[0].split(' ')[-1]  # path to sbatch.sh
            ypth = get_file_line(pth, -1).split(' ')[-1]  # path to input yaml
            inpts = load_yaml(ypth)
            output_dir = inpts['output_folder']
            name = inpts['name']

            # Save job info - state S - Submitted
            job_id = ''.join((s for s in submitted.stdout if s.isdigit()))
            self.job_history[job_id] = dict(jobid=job_id, state='S',
                                            name=name, output=output_dir)

        # Updates are slow, so don't do them often
        update = True
        if update:
            if time_module.time() - self.last_update_time > self._update_delay:
                current_jobs = self._get_slurm_info(keys)
                self.last_update_time = time_module.time()
                ended_jobs = [k for k in self.job_history
                              if k not in current_jobs]
                self._check_ended_job_status(ended_jobs)

    def _check_ended_job_status(self, jobs: List[str]) -> None:
        """
        Check the logs of the given jobs to see if the Pipeline is complete.
        """
        for j in jobs:
            logfile = os.path.join(self.job_history[j]['output'], 'log.txt')
            last_line = get_file_line(logfile, -1)

            if 'Pipeline completed.' in last_line:
                self.job_history[j]['state'] = 'C'  # Complete
            elif 'completed by Pipeline.' in last_line:
                self.job_history[j]['state'] = 'P'  # Partial
            else:
                self.job_history[j]['state'] = 'K'  # Killed

    def _check_user_jobs(self,
                         partition: str = '$GROUP',
                         user: str = '$USER'
                         ) -> int:
        """
        Checks the SLURM queue and returns the number of jobs
        """
        # Count jobs using the number of lines - does not distinguish running or not
        self.logger.info(f'Checking jobs for {user} in {partition}.')
        partition_jobs = self._get_slurm_info(['jobid'], partition, user)
        return len(partition_jobs)

    def _get_slurm_info(self,
                        keys: list = ['jobid'],
                        partition: str = '$GROUP',
                        user: str = '$USER'
                        ) -> dict:
        """
        Gets information about SLURM jobs. Wrapper for squeue.
        """
        # Dictionary to translate keys to SLURM commands
        key_to_format = dict(jobid='ArrayJobID',
                             runtime='TimeUsed',
                             cpu='cpus_per_task',
                             exitcode='exit_code',
                             state='StateCompact')
        if isinstance(keys, str):
            keys = [keys]

        # jobid is always included in order to identify jobs
        if 'jobid' not in keys:
            keys.insert(0, 'label')
        elif keys[0] != 'jobid':
            keys.remove('label')
            keys.insert(0, 'label')

        # Generate the command string and submit to shell
        format_string = ':>,'.join([key_to_format[k] for k in keys])
        command = f'squeue -p {partition} -u {user} -h --Format="{format_string}"'
        result = subprocess.run([command], shell=True, capture_output=True, text=True)

        # Parse the output string
        result = [r.rstrip().split('>') for r in result.stdout.split('\n')
                  if r != '']
        current_jobs = {r[0]: {k: v for k, v in zip(keys, r)}
                               for r in result}
        return current_jobs

    def _create_bash_script(self,
                            partition: str = '$GROUP',
                            job_name: str = 'cst',
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

        TODO:
            - Clean up
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

