from dataclasses import dataclass
from abc import ABC


@dataclass(init=False)
class Flag(ABC):
    name: str
    tooltip: str
    shortcut: str


class FlagIgnore(Flag):
    name = 'Ignore'
    tooltip = 'Ignore this in the dataset (false positive)'
    shortcut = 'x'


class FlagValidate(Flag):
    name = 'Validate'
    tooltip = 'Mark this sample as needing re-validation'
    shortcut = 'v'

