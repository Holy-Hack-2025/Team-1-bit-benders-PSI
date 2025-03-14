from dataclasses import dataclass
from typing import Tuple, List
from random import random, shuffle, normalvariate as normal, randint
from pprint import pprint

import numpy as np

from structures import *

def shuffled(l: List) -> List:
    l = l[:]
    shuffle(l)
    return l

@dataclass
class Process:
    id: int
    time: int
    allowedInputs: List[ItemType]|None
    allowedMetals: List[MetalType]|None
    targetType: ItemType
    queue: List[Item] = None
    timeLeft: int = 0

    def __post_init__(self):
        self.queue = []
        self.timeLeft = self.time 

    def step(self) -> Item|None:
        if len(self.queue) > 0:
            if self.timeLeft > 0:
                self.timeLeft -= 1
            else:
                # reset time counter
                self.timeLeft = self.time
                # update item state to next type
                if self.targetType is not None:
                    self.queue[0].type = self.targetType
                    self.queue[0].quality += normal(0, 0.1)
                    self.queue[0].deviation += normal(0, 0.1)
                    self.queue[0].hardness += normal(0, 0.1)



                # remove item and update priority queue
                out = self.queue.pop(0)
                self.queue.sort(key=lambda i: i.priority)
                return out

    def __call__(self, item: Item) -> None:
        if self.allowedInputs is not None and item.type not in self.allowedInputs:
            raise TypeError(f"Process Cannot Proceed on This Type ({item.type} not in {self.allowedInputs}).")
        if self.allowedMetals is not None and item.metal not in self.allowedMetals:
            raise TypeError(f"Process Cannot Proceed on This Metal ({item.metal} not in {self.allowedMetals}).")
        self.queue.append(item)
    
shape_block = Process(0, 3, [ItemType.RAW], None, ItemType.BLOCK)
shape_rod = Process(1, 3, [ItemType.RAW], None, ItemType.ROD)
shape_sheet = Process(2, 4, [ItemType.RAW], None, ItemType.SHEET)

machine_screw = Process(3, 2, [ItemType.BLOCK, ItemType.ROD], [MetalType.IRON, MetalType.STEEL], ItemType.SCREW)
machine_spring = Process(4, 4, [ItemType.BLOCK, ItemType.ROD], [MetalType.IRON, MetalType.STEEL], ItemType.SPRING)
machine_washer = Process(5, 2, [ItemType.BLOCK, ItemType.SHEET], [MetalType.TIN, MetalType.IRON], ItemType.WASHER)
machine_chassis = Process(6, 7, [ItemType.SHEET], [MetalType.TIN, MetalType.STEEL], ItemType.CHASSIS)

# sell = proc 7
# scrap = proc 8

processes = [shape_block, shape_rod, shape_sheet, machine_screw, machine_spring, machine_washer, machine_chassis]
processNames = ["shape_block",
                "shape_rod",
                "shape_sheet",
                "machine_screw",
                "machine_spring",
                "machine_washer",
                "machine_chassis",
                "sell",
                "scrap"]

def can_make(i: Item, o: Item) -> bool:
    if i.metal != o.metal: return False
    if o.type in [ItemType.BLOCK, ItemType.ROD, ItemType.SHEET]: return i.type == ItemType.RAW
    if o.type in [ItemType.SCREW, ItemType.SPRING]: return i.type in [ItemType.RAW, ItemType.BLOCK, ItemType.ROD]
    if o.type == ItemType.WASHER: return i.type in [ItemType.RAW, ItemType.BLOCK, ItemType.SHEET]
    if o.type == ItemType.CHASSIS: return i.type in [ItemType.RAW, ItemType.SHEET]

def item_matches(item: Item, req: Item) -> bool:
    return all([
        item.type == req.type,
        item.metal == req.metal,
        item.deviation < req.deviation,
        item.hardness > req.hardness,
        item.quality > req.quality,
        item.purity > req.purity,
        item.coef_thermalexpansion < req.coef_thermalexpansion
    ])

def heuristic_decision(inputState: InputState) -> Decision:
    item = inputState.item
    orders = inputState.requirements.items
    orders = sorted(orders, key=lambda o: o.item.priority)

    if item.deviation > .9 or item.quality < .2: return Decision(8) # scrap
    
    for order in orders:
        if item_matches(item, order.item):
            return Decision(7)
    
    for order in orders:
        for p in shuffled(processes):
            if (p.allowedInputs is None or item.type in p.allowedInputs) and (p.allowedMetals is None or item.metal in p.allowedMetals):
                return Decision(p.id)

    return Decision(8)
    

def generate_dataset(iterations: int) -> List[Tuple[InputState, Decision]]:
    orders: List[Order] = []
    items: List[Item] = []
    scrapped: List[Item] = []

    dataset: List[Tuple[InputState, Decision]] = []

    for _ in range(iterations):
        # add new order randomly
        if random() > 0.8:
            item = Item(
                shuffled(list(MetalType))[0],
                shuffled(list(ItemType))[0],
                normal(0.7, 0.1),
                normal(0.7, 0.1),
                normal(10, 3),
                random(),
                abs(normal(0, 0.1)),
                1-abs(normal(0, 0.1))
            )

            amt = randint(1, 10)

            orders += [Order(item, amt)]

        # add new item randomly
        if random() > 0.5:
            item = Item(
                shuffled(list(MetalType))[0],
                ItemType.RAW,
                normal(0.7, 0.15),
                normal(0.7, 0.15),
                abs(normal(10, 4)),
                5,
                abs(normal(0, 0.2)),
                1-abs(normal(0, 0.1))
            )

            items += [item]

        for item in items:
            # create new data entry
            state = InputState(item, Requirements.from_list(orders))
            decision = heuristic_decision(state)
            dataset += [(state, decision)]

            # process decision
            if decision.proc == 7:
                for order in orders:
                    if item_matches(item, order.item):
                        order.amount -= 1
                        break
            elif decision.proc == 8:
                scrapped += [item]
            else:
                processes[decision.proc](item)

        items = [] # empty items after processing

        # step all processes
        for process in processes:
            result = process.step()
            if result is not None: items += [result]
        
    return dataset



def human_dataset(dataset: List[Tuple[InputState, Decision]]) -> List[Tuple[InputState, Tuple[Decision, str]]]:
    ret = []

    for ds in dataset:
        ret.append((ds[0], (ds[1], processNames[ds[1].proc])))

    return ret

def print_dataset(dataset: List[Tuple[InputState, Decision]]):
    for ds in human_dataset(dataset):
        print("Input: ", end="")
        pprint(ds[0])

        print("=> ", end="")
        pprint(ds[1])

    print("\n□", dataset_stats(dataset))


def dataset_stats(dataset: List[Tuple[InputState, Decision]]):
    l = [d[1][1] for d in human_dataset(dataset)]
    d = dict.fromkeys(l, 0)
    for val in l:
        d[val] += 1
    return d


def generate_data_files(iterations = 30):
    dataset = generate_dataset(iterations)
    np.savez("./data/in", np.array([d[0].data() for d in dataset]))
    np.savez("./data/out", np.array([d[1].data() for d in dataset]))


generate_data_files()
