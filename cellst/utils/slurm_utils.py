import subprocess
import logging

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
    __name__ = 'SlurmController'
    _line_offset = 1  # Number of lines squeue returns if n_jobs == 0
    _delay = 10  # Wait between commands in sec

    def _check_user_jobs(self,
                         partition: str = '$GROUP',
                         user: str = '$USER'
                         ) -> int:
        """
        Checks the SLURM queue and returns the number of jobs
        """
        # Count jobs using the number of lines - does not distinguish running or not
        command = f'squeue -p {partition} -u {user} | wc -l'
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE)

        return int(result.stdout) - self._line_offset

    def _create_bash_script(self,
                            partition: str = '$GROUP',
                            job_name: str = 'cellst_pipe',
                            time: str = '24:00:00',
                            ntasks: str = '1',
                            cpus_per_task: str = '1',
                            mem: str = '8GB',
                            fname: str = '.temp.sh'
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

        command = 'python -m cellst.pipeline -y home/groups/sjeknic/CellST/pipeline_yamls/pipeline_yamls/F5-Site_1.yaml'
        _add_line(command, '', '')