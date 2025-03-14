from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np

class Vectorable:
    def data(self) -> np.ndarray:
        return np.array(self._data_list())

    def __convert(self, data) -> List:
        if isinstance(data, Vectorable) :
            return data._data_list()

        if isinstance(data, list):
            return sum([self.__convert(d) for d in data], [])

        return [data]

    def _data_list(self) -> List:
        values = self.__dict__
        return sum([self.__convert(values[key]) for key in values], [])
        

class EnumVec(Vectorable, Enum):
    def _data_list(self):
        v = [0] * len(self.__class__)
        v[self.value] = 1
        return v


class MetalType(EnumVec):
    TIN = 0
    STEEL = 1
    IRON = 2

    @staticmethod
    def parse(data):
        return MetalType(list(data).index(1))

class ItemType(EnumVec):
    SCREW = 0
    SPRING = 1
    WASHER = 2
    PIN = 3
    CHASSIS = 4

    @staticmethod
    def parse(data):
        return ItemType(list(data).index(1))

class ItemField(EnumVec):
    METAL = 0
    TYPE = 1
    PURITY = 2
    HARDNESS = 3
    COEF_THERMALEXPANSION = 4
    PRIORITY = 5
    DEVIATION = 6
    QUALITY = 7

    @staticmethod
    def parse(data):
        return ItemField(list(data).index(1))

class ConstraintType(EnumVec):
    MIN = 0
    MAX = 1
    WHITHIN = 2
    EQ = 3

    @staticmethod
    def parse(data):
        return ConstraintType(list(data).index(1))

@dataclass
class Item(Vectorable):
    metal: MetalType
    type: ItemType

    purity: float
    hardness: float
    coef_thermalexpansion: float
    priority: float
    deviation: float = 0
    quality: float = 1


    @staticmethod
    def parse(data):
        return Item(MetalType.parse(data[[i for i in range(len(MetalType))]]),
                    ItemType.parse(data[[i + len(MetalType) for i in range(len(ItemType))]]),
                    data[len(MetalType) + len(ItemType)],
                    data[len(MetalType) + len(ItemType) + 1],
                    data[len(MetalType) + len(ItemType) + 2],
                    data[len(MetalType) + len(ItemType) + 3],
                    data[len(MetalType) + len(ItemType) + 4],
                    data[len(MetalType) + len(ItemType) + 5],
                    )

@dataclass
class Order(Vectorable):
    items: List[Item]

    @staticmethod
    def parse(data):
        size = len(Item(MetalType.IRON, ItemType.CHASSIS, 0, 0, 0, 0).data())

        split = []
        for i in range(0, len(data), size):
            split.append(data[i:i + size])

        return Order([Item.parse(d) for d in split])
