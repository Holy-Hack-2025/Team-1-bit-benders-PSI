from dataclasses import dataclass

from data.format.items import Vectorable
from data.format.factory import Process


@dataclass
class Result(Vectorable):
    proc: Process

    @staticmethod
    def parse(data):
        return Result(Process.parse(data))