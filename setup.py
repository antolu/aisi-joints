"""
Dynamic dependencies implementation from CERN accwidgets
"""
from typing import List, Dict

from setuptools import setup
import os
from os import path

from pathlib import Path

try:
    import pyqt5ac

    # generate PyQt UI modules
    try:
        # compile pyqt files
        HERE = path.split(path.abspath(__file__))[0]
        pyqt5ac.main(
            uicOptions='--from-imports', force=False, initPackage=True,
            ioPaths=[
                [path.join(HERE, 'resources/ui/*.ui'),
                 path.join(HERE,
                           'aisi_joints/ui/generated/%%FILENAME%%_ui.py')],
                [path.join(HERE, 'resources/*.qrc'),
                 path.join(HERE, 'aisi_joints/ui/generated/%%FILENAME%%_rc.py')]
            ]
        )
    except (PermissionError, OSError):
        pass
except ImportError:
    print('Could not import pyqt5ac. Skipping regeneration of resource files.')

FOUND_USER_DEPS_FILES = []
FOUND_DEV_DEPS_FILES = []
# Files to search for -> requirements.txt = minimal deps to run, dev_requirements = deps for running tests...
USR_DEPS_FILENAME = "requirements.txt.txt"
DEV_DEPS_FILENAME = "dev_requirements.txt"
DEV_DEPS_MAP_KEY = "testing"
CURRENT_FILE_LOCATION = path.abspath(path.dirname(__file__))

PACKAGES = ["aisi_joints"]
INSTALL_REQUIRES: List[str] = []
EXTRA_REQUIRES: Dict[str, List[str]] = {DEV_DEPS_MAP_KEY: [], 'all': []}

print(f"Search for files {USR_DEPS_FILENAME} and {DEV_DEPS_FILENAME} recursively, "
      f"starting from {CURRENT_FILE_LOCATION}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start Search ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for package in PACKAGES:
    folder_to_search = (
            CURRENT_FILE_LOCATION
            + ("" if CURRENT_FILE_LOCATION[-1] == path.sep else path.sep)
            + package
    )
    print(f"Search folder:                     {folder_to_search}")
    for root, _, files in os.walk(
            folder_to_search,
            onerror=(lambda err, folder=folder_to_search: print(f"{folder} not found.")),  # type: ignore
    ):
        for file in files:
            if DEV_DEPS_FILENAME == file:
                print(f"Found developer requirements.txt file: {path.join(root, file)}")
                FOUND_DEV_DEPS_FILES.append(path.join(root, file))
            elif USR_DEPS_FILENAME == file:
                print(f"Found user requirements.txt file:      {path.join(root, file)}")
                FOUND_USER_DEPS_FILES.append(path.join(root, file))


for usr_dep_file in FOUND_USER_DEPS_FILES:
    submodule = path.split(path.split(usr_dep_file)[0])[1]

    with open(path.join(usr_dep_file), "r") as f:
        deps = f.read().strip().split("\n")
        print(f"Collecting user dependencies:      {deps}")
        # INSTALL_REQUIRES += deps

        EXTRA_REQUIRES[submodule] = deps
        EXTRA_REQUIRES['all'] += deps

for dev_dep_file in FOUND_DEV_DEPS_FILES:
    with open(path.join(dev_dep_file), "r") as f:
        deps = f.read().strip().split("\n")
        print(f"Collecting developer dependencies: {deps}")
        EXTRA_REQUIRES["testing"] += deps

EXTRA_REQUIRES["linting"] = [
    "mypy~=0.720",
    "pylint>=2.3.1&&<3",
    "pylint-unittest>=0.1.3&&<2",
    "flake8>=3.7.8&&<4",
    "flake8-quotes>=2.1.0&&<3",
    "flake8-commas>=2&&<3",
    "flake8-colors>=0.1.6&&<2",
    "flake8-rst>=0.7.1&&<2",
    "flake8-breakpoint>=1.1.0&&<2",
    "flake8-pyi>=19.3.0&&<20",
    "flake8-comprehensions>=2.2.0&&<3",
    "flake8-builtins-unleashed>=1.3.1&&<2",
    "flake8-blind-except>=0.1.1&&<2",
    "flake8-bugbear>=19.8.0&&<20",
]

curr_dir: Path = Path(__file__).parent.absolute()

with curr_dir.joinpath("README.md").open() as f:
    long_description = f.read()


setup(extras_require=EXTRA_REQUIRES)
