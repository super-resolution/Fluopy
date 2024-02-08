"""
Module abstract_classes
"""
from abc import ABC
from dataclasses import dataclass


@dataclass
class FluorophoreData(ABC):
    """
    Abstract class, intended to be subclassed by fluorophore dataclasses that represent real fluorophores.
    """
    pass
