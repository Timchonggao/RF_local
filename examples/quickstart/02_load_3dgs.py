from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter


@dataclass
class Load3DGS(Task):

    load: Path = ...

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        print(f'Result: #Gaussians = {model.gaussians.shape[0]}')


if __name__ == '__main__':
    Load3DGS(cuda=0).run()
