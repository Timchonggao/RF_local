# Training

Each training task consists of the following key concepts:

1. Dataset
  Dataset provides input data and meta data required of the task. For example, in the novel view synthesis tasks, input data could be a set of calibrated cameras from multiple viewpoints and meta data can be the underlying mesh model if it is a synthetic dataset. Moreover, Dataset will provide grouth truth output data for supervision, (e.g. GT RGB images in novel view synthesis).

2. Experiment
  Experiment is a special module aiming to record training information like checkpoints of the trained model, text logs or image visualizations. Also, it supports loading the pretrained model from a previous experiment. All the IO operations as well as the file system interaction will be handled by Experiment.

3. Model
  Model is the parameterized algorithm that will be trained in the task. Besides the must of being a nn.Module instance, there is no other assumption about Model. All the detailed interaction with the dataset or optimizers are handled in Trainer.

4. Trainer
  Trainer is the key module that handles how the data should be fed into the model, and how the loss are computed from the model output and the provided supervision. Also, Trainer determines how to conduct visualization during the training.

