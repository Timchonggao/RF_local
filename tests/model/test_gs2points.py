from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter


@dataclass
class GS2Points(Task):

    load: Path = ...

    output: Path = ...

    num_samples: int = 100000

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        model.gaussians.as_points(self.num_samples).export(self.output)


if __name__ == '__main__':
    GS2Points().run()
