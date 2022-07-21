"""
Script to install latest Tensorflow Object Detection API
"""
import glob
import re
import os
import platform
import shutil
import stat
import sys
import sysconfig
import zipfile
import urllib.request

import requests
import subprocess

PROTOBUF_URL = 'https://api.github.com/repos/protocolbuffers/protobuf/releases/latest'
PROTOBUF_ZIP = 'protobuf.zip'
PROTOBUF_DIR = 'protobuf'

MODELS_URL = 'https://github.com/tensorflow/models.git'


def get_latest_protobuf(platform_id: str):
    api_json = requests.get(PROTOBUF_URL).json()

    regex = re.compile(rf'.*{platform_id}.*')

    browser_url = None
    for idx, data in enumerate(api_json['assets']):
        if regex.match(data['name']):
            browser_url = data['browser_download_url']

    if browser_url is None:
        raise ValueError(f'Could not find protobuf release for {platform_id}.')

    print(f'Found protobuf release at {browser_url}.')

    return browser_url


def get_od_repo(clone_to: str) -> str:
    command = f'git clone https://github.com/tensorflow/models.git {clone_to}'
    print(f'Running command {command}')
    result = subprocess.run(command.split(' '))
    if result.returncode != 0:
        if os.path.isdir('models'):
            print(f'WARNING: directory {clone_to} already exists.')
        else:
            sys.exit(result.returncode)

    return clone_to


def compile_protobuf(protobuf_exec: str):
    if not os.path.isfile(protobuf_exec):
        if os.path.isfile(protobuf_exec + '.exe'):  # Windows check
            protobuf_exec += '.exe'
            prefix = '.\\'
        else:
            raise ValueError
    else:
        prefix = './'

    st = os.stat(protobuf_exec)
    os.chmod(protobuf_exec, st.st_mode | stat.S_IEXEC)
    print(f'Compiling protobuf messages using binary at {protobuf_exec}.')
    command = f'{prefix}{protobuf_exec} ' \
              f'{" ".join(glob.glob("object_detection/protos/*.proto"))} --python_out=.'

    print(f'Running command {command} in {os.getcwd()}')
    result = subprocess.run(command.split(' '))
    if result.returncode != 0:
        sys.exit(result.returncode)


def install_object_detection():
    cwd = os.getcwd()

    os.chdir('object_detection/packages/tf2')
    command = 'python -m pip install .'

    print(f'Running command {command} in {os.getcwd()}')
    result = subprocess.run(command.split(' '))

    if result.returncode != 0:
        sys.exit(result.returncode)

    os.chdir(cwd)


def main(cleanup: bool = True):
    cwd = os.getcwd()

    get_od_repo('models')
    os.chdir('models/research')

    system = platform.system()
    machine = sysconfig.get_platform().split("-")[-1].lower()

    if system == 'Windows':
        platform_id = 'win64'
    elif system == 'Linux':
        platform_id = 'linux-x86_64'
    elif system == 'Darwin':
        if machine == 'x86_64':
            platform_id = 'osx-x86_64'
        elif machine == 'arm64':
            platform_id = 'osx-universal_binary'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f'Script does not support '
                                  f'OS {system}.')

    protobuf_url = get_latest_protobuf(platform_id)

    print(f'Downloading protobuf to {PROTOBUF_ZIP}')
    urllib.request.urlretrieve(protobuf_url, PROTOBUF_ZIP)

    print(f'Unzipping protobuf to {PROTOBUF_DIR}')
    with zipfile.ZipFile(PROTOBUF_ZIP, 'r') as zip_ref:
        zip_ref.extractall(PROTOBUF_DIR)

    compile_protobuf(os.path.join(PROTOBUF_DIR, 'bin', 'protoc'))

    install_object_detection()

    if cleanup:
        shutil.rmtree(PROTOBUF_DIR)
        shutil.rmtree(PROTOBUF_ZIP)
        os.chdir(cwd)
        shutil.rmtree('models')


if __name__ == '__main__':
    main()
