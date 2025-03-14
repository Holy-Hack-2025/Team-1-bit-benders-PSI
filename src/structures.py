from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, get_origin
import numpy as np

class Vectorable:
    def data(self) -> np.ndarray:
        return np.array(self._data_list())

    def __convert(self, data) -> List:
        if isinstance(data, Vectorable) :
            return data._data_list()

        if isinstance(data, list):
            return sum([self.__convert(d) for d in data], [])
        
        if isinstance(data, tuple):
            return sum([self.__convert(d) for d in data], [])

        return [data]

    def _data_list(self) -> List:
        values = self.__dict__
        return sum([self.__convert(values[key]) for key in values], [])
    
def len_data(vectorable: Vectorable) -> int:
    fields = vectorable.__annotations__
    return sum([ \
                sum([len_data(item) for item in field.__args__]) if get_origin(field) is tuple \
                    else len(field) if issubclass(field, Enum) \
                    else len_data(field) if issubclass(field, Vectorable) \
                    else 1 \
                    for field in fields.values()] \
            )

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
    def parse(data: np.ndarray):
        return MetalType(list(data).index(1))

class ItemType(EnumVec):
    RAW = 0
    SHEET = 1
    BLOCK = 2
    ROD = 3
    SCREW = 4
    SPRING = 5
    WASHER = 6
    CHASSIS = 7

    @staticmethod
    def parse(data: np.ndarray):
        return ItemType(list(data).index(1))

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
    def parse(data: np.ndarray):
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
    item: Item
    amount: int

    @staticmethod
    def parse(data: np.ndarray):
        return Order(Item.parse(data[:-1]), int(data[-1]))


@dataclass
class Requirements(Vectorable):
    items: Tuple[Order, Order, Order, Order, Order]

    @staticmethod
    def parse(data: np.ndarray):
        size = len_data(Order)

        split = []
        for i in range(0, len(data), size):
            split.append(data[i:i + size])

        return Requirements(tuple(Order.parse(d) for d in split))
    
    @staticmethod
    def from_list(orders: List[Order]):
        orders = sorted(orders, key=lambda o: o.item.priority)
        if len(orders) < 5: orders += [Order(Item(MetalType.IRON, ItemType.BLOCK, 0, 0, 0, 0), 0)] * ( 5 - len(orders) )
        return Requirements(tuple(orders[:5]))

@dataclass
class InputState(Vectorable):
    item: Item
    requirements: Requirements

    @staticmethod
    def parse(data: np.ndarray):
        itemlen = len_data(Item)
        return InputState(Item.parse(data[:itemlen]), Requirements.parse(data[itemlen:]))


@dataclass
class Decision(Vectorable):
    proc: int

    def data(self):
        return self.proc

    @staticmethod
    def parse(data: np.ndarray):
        return Decision(list(data).index(1))
    
