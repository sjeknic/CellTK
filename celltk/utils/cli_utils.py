import argparse


class CLIParser():

    def __init__(self, name: str = None) -> None:
        # Select the parser
        try:
            parse_method = getattr(self, f'{name}_parser')
        except AttributeError:
            raise AttributeError('Could not find command line parser '
                                 f'for input {name}.')

        # Create the parser
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        self.cli_parser = parse_method(parser)

    def get_command_line_inputs(self) -> argparse.Namespace:
        return self.cli_parser.parse_args()

    def orchestrator_parser(self,
                            parser: argparse.ArgumentParser
                            ) -> argparse.ArgumentParser:
        parser.add_argument(
            '--pipelines', '-l',  # TODO: Might need to think of better shortcuts
            default=None,
            type=str,
            help='Path to directory containing the pipeline yamls.'
        )
        parser.add_argument(
            '--operations', '-o',
            default=None,
            type=str,
            help='YAML file containing operation parameters.'
        )

        # Parameterize the Orchestrator
        parser.add_argument(
            '--parent', '-p',
            default=None,
            type=str,
            help='Path to directory containing folders of images.'
        )
        # TODO: Add option to pass relative path to output directory
        parser.add_argument(
            '--output',
            type=str,
            default='output',
            help='Path to directory to save images in.'
        )
        parser.add_argument(
            '--image', '-i',
            default=None,
            type=str,
            help='Name of directory containing images.'
        )
        parser.add_argument(
            '--mask', '-m',
            default=None,
            type=str,
            help='Name of directory containing masks.'
        )
        parser.add_argument(
            '--track', '-t',
            default=None,
            type=str,
            help='Name of directory containing tracks.'
        )
        parser.add_argument(
            '--array', '-a',
            default=None,
            type=str,
            help='Name of directory containing arrays.'
        )
        parser.add_argument(
            '--extension',
            default='tif',
            type=str,
            help='Extension of the images to look for. tif by default.'
        )
        parser.add_argument(
            '--match-str',
            default=None,
            type=str,
            help='If set, image files or folder must contain match-str'
        )
        parser.add_argument(
            '--name',
            default='experiment',
            type=str,
            help='Name of experiment to use.'
        )
        parser.add_argument(
            '--cond-map',
            default=None,
            type=str,
            help='Path to YAML file containing condition map.'
        )

        # Running Orchestrator
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='If set, will overwrite contents of output folder. Otherwise makes new folder.'
        )
        parser.add_argument(
            '--no-save-pipe-yamls',
            action='store_true',
            help='If set, will NOT write YAML files for each pipeline to output/pipeline_yamls.'
        )
        parser.add_argument(
            '--no-save-master-df',
            action='store_true',
            help='If set, will NOT save dataframe for the whole experiment.'
        )
        parser.add_argument(
            '--no-log',
            action='store_true',
            help='If set, outputs are logged only to the console.'
        )
        parser.add_argument(
            '--run',
            action='store_true',
            help='If set, will run the Orchestrator and save results in output.'
        )
        parser.add_argument(
            '--njobs', '-n',
            default=1,
            type=int,
            help='Number of parallel Pipelines to run simultaneously.'
        )

        # Add SLURM arguments
        controller = parser.add_subparsers(help='Options for SLURM')
        slurm_parser = controller.add_parser('slurm')

        # SLURM - Job control
        slurm_parser.add_argument(
            '--partition', '-p',
            default='$GROUP',
            nargs='*',
            type=str,
            help='Name of the partition to submit jobs to.'
        )
        # NOTE: Right now user is only used to get num jobs in queue
        slurm_parser.add_argument(
            '--user', '-u',
            default='$USER',
            type=str,
            help='Name of the user submitting the jobs.'
        )
        slurm_parser.add_argument(
            '--time', '-t',
            default='24:00:00',
            type=str,
            help='Max time for job in HH:MM:SS.'
        )
        # TODO: Might want to change later to include other cpu options
        slurm_parser.add_argument(
            '--cpus', '-c',
            default=2,
            type=int,
            help='Number of CPUs to request for the job.'
        )
        slurm_parser.add_argument(
            '--mem', '-m',
            default='8GB',
            type=str,
            help='Amount of memory to request for the job.'
        )
        slurm_parser.add_argument(
            '--job-name', '-n',
            default='celltk',
            type=str,
            help='Name of the job submission.'
        )
        # TODO: Not sure about this. Might be Sherlock specific
        slurm_parser.add_argument(
            '--modules',
            default=None,
            type=str,
            help='Name of module set to load.'
        )

        # SLURM - Queue control
        slurm_parser.add_argument(
            '--maxjobs', '-x',
            default=1,
            nargs='*',
            type=int,
            help='Max jobs allowed in queue at any one time.'
        )
        slurm_parser.add_argument(
            '--maxsubs', '-s',
            default=0,
            type=int,
            help='Total job submissions allowed. If 0, no limit.'
        )

        return parser