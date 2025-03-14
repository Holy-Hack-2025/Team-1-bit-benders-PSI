from dataclasses import dataclass
from src.data.format.types import *

@dataclass
class Process(Vectorable):
    input: Item
    output: Item

    @staticmethod
    def parse(data):
        size = len(Item(MetalType.IRON, ItemType.CHASSIS, 0, 0, 0, 0).data())

        split = []
        for i in range(0, len(data), size):
            split.append(data[i:i + size])

        return Process(Item.parse(data[0:size]), Item.parse(data[size:2 * size]))