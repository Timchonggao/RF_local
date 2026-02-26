from dataclasses import dataclass

from rfstudio.engine.task import Task
from rfstudio.utils.lazy_module import _test, tcnn


@dataclass
class Test(Task):

    def run(self) -> None:
        _test # ok
        tcnn.free_temporary_memory # ok
        _test.something() # fail

if __name__ == '__main__':
    Test(cuda=0).run()
