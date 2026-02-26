from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter


@dataclass
class Test3DGS(Task):

    load: Path = ...

    experiment: Experiment = Experiment(
        name='test_3dgs',
        output_dir=Path('exports')
    )

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        dataset = train_task.dataset
        trainer = train_task.trainer

        with torch.no_grad():
            model.train(False)
            losses = []
            psnrs = []
            for inputs, gt_outputs, indices in dataset.get_train_iter(1, shuffle=False, infinite=False):
                loss, metrics, visualization = trainer.step(
                    model,
                    inputs,
                    gt_outputs,
                    training=False,
                    visual=True
                )
                self.experiment.dump_image('train-view', index=indices[0], image=visualization)
                losses.append(loss.item())
                psnrs.append(metrics['psnr'].item())
            print("Eval: loss={:.5f} psnr={:.3f}".format(
                sum(losses) / len(losses),
                sum(psnrs) / len(psnrs)
            ))

if __name__ == '__main__':
    Test3DGS(cuda=0).run()
