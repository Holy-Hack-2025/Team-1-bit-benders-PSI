from dataclasses import dataclass

from data.format.items import Vectorable

@dataclass
class Result(Vectorable):
    proc: int

    def data(self):
        v = [0] * 20
        v[self.proc] = 1
        return v

    @staticmethod
    def parse(data):
        return Result(list(data).index(1))