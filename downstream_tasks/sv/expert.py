import math
import os
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .dataset import Verification_Dataset, Classification_Dataset
# from .model import Model
from .model import AMSoftmaxLoss, AAMSoftmaxLoss, SoftmaxLoss, UtteranceExtractor
from .utils import EER

def get_ddp_sampler(dataset: Dataset, epoch: int):
    """
    This function will create a DistributedSampler if DDP is initialized,
    and will just return None if DDP is not initialized.
    """
    if is_initialized():
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(epoch)
    else:
        sampler = None
    return sampler


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self,
        preprocess,
        preprocess_audio,
        preprocess_video,
        upstream_dim,
        downstream_expert,
        expdir,
        **kwargs,
    ):
        """
        Your dataset should take two preprocessing transform functions,
        preprocess_audio and preprocess_video as input.

        These two functions will be defined by the upstream models, and
        will transform raw waveform & video frames into the desired
        format of the upstream model.

        They take two arguments, the input audio/video Tensor, and the
        audio sample rate/video frame rate, respectively.

        Optionally, if you wish to obtain raw data for testing purposes,
        you may also specify these functions to be None, and return the
        raw data when the functions are not defined.
        Args:
            preprocess_audio: function
                Defined by specified upstream model, transforms raw waveform into
                desired input format.
                Takes two arguments, input audio Tensor, and audio sample rate.

            preprocess_video: function
                Defined by specified upstream model, transforms raw video frames
                into desired input format.
                Takes two arguments, input video Tensor, and video frame rate.

            upstream_dim: int
                Different upstream models will give different representation dimension
                You might want to first project them to the same dimension

            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you cab save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.

                Note. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()

        # config
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        # dataset
        train_config = {
            "split": "train",
            "dataroot": self.datarc["file_path"],
        }
        self.train_dataset = Classification_Dataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'], 
            **train_config
        )

        dev_config = {
            "split": "valid",
            "dataroot": self.datarc["file_path"],
        }
        self.dev_dataset = Classification_Dataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'],
            **dev_config
        )

        test_config = {
            "dataroot": self.datarc["file_path"],
        }
        self.test_dataset = Verification_Dataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'],
            **test_config
        )

        # module
        self.connector = nn.Linear(self.upstream_dim, self.modelrc["input_dim"])

        # # downstream model
        # agg_dim = self.modelrc["module_config"][self.modelrc["module"]].get(
        #     "agg_dim", self.modelrc["input_dim"]
        # )

        # ModelConfig = {
        #     "input_dim": self.modelrc["input_dim"],
        #     "agg_dim": agg_dim,
        #     "agg_module_name": self.modelrc["agg_module"],
        #     "module_name": self.modelrc["module"],
        #     "hparams": self.modelrc["module_config"][self.modelrc["module"]],
        #     "utterance_module_name": self.modelrc["utter_module"],
        # }
        # # downstream model extractor include aggregation module
        # self.model = Model(**ModelConfig)

        # SoftmaxLoss or AMSoftmaxLoss
        objective_config = {
            "speaker_num": self.train_dataset.speaker_num,
            "hidden_dim": self.modelrc["input_dim"],
            **self.modelrc["LossConfig"][self.modelrc["ObjectiveLoss"]],
        }

        self.objective = eval(self.modelrc["ObjectiveLoss"])(**objective_config)

        # utils
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.eval_metric = EER

        self.register_buffer("best_score", torch.zeros(1))
        self.register_buffer("best_eer", torch.ones(1) * 100)

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            split: string
                'train'
                    will always be called before the training loop

                'dev', 'test', or more
                    defined by the 'eval_dataloaders' field in your downstream config
                    these will be called before the evaluation loops during the training loop

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [(wav1,vid1), (wav2,vid2), ...], your_other_contents1, your_other_contents2, ...
        """

        if split == "train":
            return self._get_train_dataloader(self.train_dataset, epoch)
        elif split == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset, epoch: int):
        sampler = get_ddp_sampler(dataset, epoch)
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=None,
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def forward(self, split, features, others, basename, labels=None, records=None, **kwargs):
        """
        Args:
            split: string
                'train'
                    when the forward is inside the training loop

                'dev', 'test' or more
                    when the forward is inside the evaluation loop

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            others:
                During train and dev, others are labels
                During test, others are paths

            labels:
                the speaker labels
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
                
        features_pad = pad_sequence(features, batch_first=True)
        # attention_mask = [torch.ones((feature.shape[0])) for feature in features]

        # attention_mask_pad = pad_sequence(attention_mask, batch_first=True)
        # attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)

        if split == "train" or split == "dev":
            labels = others
            agg_vec = features_pad.mean(dim=1)
            labels = torch.LongTensor(labels).to(features_pad.device)

            loss, predicted = self.objective(agg_vec, labels)
            predicted_classid = predicted.max(dim=-1).indices
            
            records["loss"].append(loss.item())
            records["acc"] += (
                (predicted_classid == labels).view(-1).cpu().float().tolist()
            )

            return loss
            
        elif split == "test":
            agg_vec = features_pad.mean(dim=1)
            agg_vec = torch.nn.functional.normalize(agg_vec, dim=-1)
            utt_name = others

            for idx in range(len(agg_vec)):
                records[utt_name[idx]] = agg_vec[idx].cpu().detach()
            return torch.tensor(0)

    # interface
    def log_records(
        self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev', 'test' or more:
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        save_names = []

        if split == "train":
            loss = torch.FloatTensor(records["loss"]).mean().item()
            logger.add_scalar(f"voxceleb2/{split}-loss", loss, global_step=global_step)

        elif split == "dev":
            for key, values in records.items():
                average = torch.FloatTensor(values).mean().item()
                logger.add_scalar(
                    f"voxceleb2/{split}-{key}", average, global_step=global_step
                )
                if split == "dev" and key == "acc" and average > self.best_score:
                    self.best_score = torch.ones(1) * average
                    save_names.append(f"{split}-best.ckpt")

        elif split == "test":
            trials = self.test_dataset.pair_table
            labels = []
            scores = []
            pair_names = []
            for label, name1, name2 in trials:
                labels.append(label)
                score = self.score_fn(records[name1], records[name2]).numpy()
                scores.append(score)
                pair_names.append(f"{name1} {name2}")
            eer, *others = self.eval_metric(
                np.array(labels, dtype=int), np.array(scores)
            )
            logger.add_scalar(f"voxceleb2/{split}-EER", eer, global_step=global_step)

            if split == "test" and eer < self.best_eer:
                self.best_eer = torch.ones(1) * eer
                save_names.append(f"{split}-best.ckpt")

            with open(Path(self.expdir) / f"{split}_predict.txt", "w") as file:
                line = [f"{name} {score}\n" for name, score in zip(pair_names, scores)]
                file.writelines(line)

            with open(Path(self.expdir) / f"{split}_truth.txt", "w") as file:
                line = [f"{name} {score}\n" for name, score in zip(pair_names, labels)]
                file.writelines(line)

        return save_names
