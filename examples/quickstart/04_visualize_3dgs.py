from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter
from rfstudio.visualization import vis_3dgs


@dataclass
class Vis3DGS(Task):

    load: Path = ...

    port: int = ...

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        vis_3dgs(model.gaussians, port=self.port)


if __name__ == '__main__':
    Vis3DGS(port=6789).run()
