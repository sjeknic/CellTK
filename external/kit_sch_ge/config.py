from pathlib import Path


def get_project_path():
    return Path(__file__).parent


def get_data_path():
    project_path = get_project_path()
    parent_dir = project_path.parent
    return parent_dir / 'data'


def get_results_path():
    project_path = get_project_path()
    parent_dir = project_path.parent
    return parent_dir / 'results'

# get string path
def string_path(path_arg):
    if not isinstance(path_arg, str):
        if hasattr(path_arg, 'as_posix'):
            path_arg = path_arg.as_posix()
        else:
            raise TypeError('Cannot convert variable to string path')
    else:
        path_arg = path_arg.replace('\\', '/')
    return path_arg

image_formats = ('bmp', 'jpeg', 'tif', 'png', 'tiff')

