from rfstudio.data import MultiViewDataset
from rfstudio.engine.train import Experiment, TrainTask
from rfstudio.model import GSplatter
from rfstudio.trainer import GSplatTrainer

std_train_task = TrainTask(
    dataset=MultiViewDataset(
        path=...,
    ),
    model=GSplatter(
        background_color='black',
        sh_degree=0,
        prepare_densification=True,
    ),
    experiment=Experiment(name=...),
    trainer=GSplatTrainer(
        num_steps=30000,
        batch_size=1,
        num_steps_per_val=300,
        num_steps_per_save=6000,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

if __name__ == '__main__':
    std_train_task.run()
