from os import path
from .._utils.extras_require import check_dependencies

HERE = path.split(__file__)[0]

check_dependencies(path.join(HERE, 'requirements.txt'))

