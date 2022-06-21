import importlib.util
from os import path


def check_dependencies(requirements_file: str):
    submodule = path.split(path.split(requirements_file)[0])[1]

    with open(requirements_file, 'r') as f:
        dependencies = f.readlines()

    dependencies = [o.replace('-', '_').strip() for o in dependencies]

    for dep in dependencies:
        if not importlib.util.find_spec(dep):
            raise ModuleNotFoundError(
                f'Could not import dependency {dep} for submodule '
                f'{submodule}. \n'
                f'To use aisi_joints.{submodule}, '
                f'make sure you have installed aisi_joints with '
                f'the correct extras.'
                f'For instance `pip install aisi_joints[{submodule}]`.'
            )
