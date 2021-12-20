import subprocess


class SlurmController():
    _line_offset = 1  # Number of lines squeue returns if n_jobs == 0
    _delay = 10  # Wait between commands in sec

    def __init__(self) -> None:
        pass

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
                            uid: str = '$USER',
                            job_name: str = 'cellst_pipe',
                            time: str = '24:00:00',
                            ntasks: str = '1',
                            cpus_per_task: str = '1',
                            mem: str = '8GB'
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
        params = {l: v for l, v in locals().items()
                  if l != 'self' and l != '_add_line'}

        # Start with bash
        string = '#!/bin/bash'
        for p, val in params.items():
            pname = p.replace('_', '-')
            to_add = f'--{pname}={val}'
            string = _add_line(string, to_add)

        string = _add_line(string, '', '', '')

        # TODO: Add the real python command here
        mods = 'module load cellst376'
        string = _add_line(string, mods, '', '')

        command = 'python -m cellst.pipeline -y /home/users/sjeknic/CellST/test_out_orch/pipeline_yamls/F3-Site_3.yaml'
        string = _add_line(string, command, '', '')

        subprocess.run([string], shell=True)