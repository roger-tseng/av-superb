# Guidelines for Contributors

1. Create an branch for each of your respective models/tasks.
2. Implement your respective models/tasks following the `example` folders.
3. Push your progress to our repository before our meetings
3. (TEMPORARY) Before the interface connecting upstream models and downstream models has been finished, contributors may use the [S3PRL toolkit](https://github.com/s3prl/s3prl) to test the audio part of your models/tasks.

## Upstream Models

Contributors of upstream models should implement a class following `upstream_models/example/expert.py` with the following functions:

- `preprocessing_audio` & `preprocessing_video`: <br/>
    takes an audio/video torch.Tensor and turns in into the input format of the model.
- `forward`: <br/>
    - **input**: list of preprocessed audio-video tuples `[(wav1,vid1), (wav2,vid2), ...]` 
    - **output**: a dictionary where each key's corresponding value is either a padded sequence in torch.FloatTensor or a list of padded sequences, each in torch.FloatTensor. Every padded sequence is in the shape of (batch_size, max_sequence_length_of_batch, hidden_size). At least a key `"hidden_states"` is available, which is a list.

## Downstream Tasks

Contributors of downstream tasks should modify three `.py` files following `downstream_tasks/example`.

1. `dataset.py` loads the dataset for your task. <br/>
    This file should be modified to load audio waveforms, video frames, and labels of your respective datasets.
2. `model.py` contains the small probing model that takes upstream model features as input, and performs the downstream task. <br/>
    Contributors should modify model architecture of the probing model. <br/>
    Architecture of the probing model can be flexible, as long as it suits the task and remains reasonably small. (e.g mean-pooling and/or a Linear layer, a LSTM layer etc.) <br/>
    We will discuss the design choices in the meeting.
3. `expert.py` contains the remaining code for training/testing, including creating dataloaders, logging with `tensorboard`, etc. 
    Contributors should modify the `__init__` function and `forward` function to suit your design.

