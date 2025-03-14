from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np

class Vectorable:
    def data(self) -> np.ndarray:
        return np.array(self._data_list())

    def _data_list(self) -> List:
        values = self.__dict__
        return sum([values[key]._data_list() if isinstance(values[key], Vectorable) else values[key] for key in values], [])
        

class EnumVec(Vectorable, Enum):
    def data(self):
        v = np.zeros(len(self.__class__))
        v[self.value] = 1
        return v

class MetalType(EnumVec):
    TIN = 0
    STEEL = 1
    IRON = 2

class ItemType(EnumVec):
    SCREW = 0
    SPRING = 1
    WASHER = 2
    PIN = 3
    CHASSIS = 4

class ItemField(EnumVec):
    METAL = 0
    TYPE = 1
    PURITY = 2
    HARDNESS = 3
    COEF_THERMALEXPANSION = 4
    PRIORITY = 5
    DEVIATION = 6
    QUALITY = 7

class ConstraintType(EnumVec):
    MIN = 0
    MAX = 1
    WHITHIN = 2
    EQ = 3

@dataclass
class Item(Vectorable):
    metal: MetalType
    type: ItemType

    purity: float
    hardness: float
    coef_thermalexpansion: float
    priority: int
    deviation: float
    quality: float


@dataclass
class Constaint(Vectorable):
    field: ItemField
    type: ConstraintType
    value: float | int | MetalType | ItemType

@dataclass
class ItemConstaints(Vectorable):
    constraints: List[Constaint]

@dataclass
class Order(Vectorable):
    items: List[Constaint]

    