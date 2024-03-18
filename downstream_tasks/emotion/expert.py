"""
Custom class for training/testing code for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/expert.py
"""
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, random_split

from .dataset import IEMOCAPDataset, collate_fn
from .model import *

def get_ddp_sampler(dataset: IEMOCAPDataset, epoch: int):
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

    def __init__(self, preprocess, preprocess_audio,
        preprocess_video, upstream_dim, downstream_expert, expdir, **kwargs,):
        """
        Args:
            upstream_dim: int
                Different upstream models will give different representation dimension
                You might want to first project them to the same dimension
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml
            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.
            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.
                Note. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]  # config for dataset
        self.modelrc = downstream_expert["modelrc"]  # config for model
        
        DATA_ROOT = self.datarc["iemocap_root"]
        meta_data = f"{DATA_ROOT}/meta_data"
        
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"
        
        train_path = os.path.join(meta_data, self.fold.replace('fold', 'Session'), 'train_meta_data.json')
        test_path = os.path.join(meta_data, self.fold.replace('fold', 'Session'), 'test_meta_data.json')
        
        
        dataset = IEMOCAPDataset(DATA_ROOT, train_path, preprocess, preprocess_audio, preprocess_video, upstream=kwargs['upstream'], pooled_features_path=kwargs['pooled_features_path'], upstream_feature_selection=kwargs['upstream_feature_selection'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]

        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)
        self.test_dataset = IEMOCAPDataset(DATA_ROOT, test_path, preprocess, preprocess_audio, preprocess_video, upstream=kwargs['upstream'], pooled_features_path=kwargs['pooled_features_path'], upstream_feature_selection=kwargs['upstream_feature_selection'])

        self.connector = nn.Linear(upstream_dim, self.modelrc["input_dim"])
        self.model = Model(
            **self.modelrc
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer("best_score", torch.zeros(1))

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
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=collate_fn,
        )

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
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
            your_other_contents1, ... :
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
        
        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        predicted = self.model(features)

        utterance_labels = labels
        labels = torch.LongTensor(utterance_labels).to(features.device)        
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices

        records["loss"].append(loss.item())
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
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
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{split}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{split} {key}: {average}")
                    f.write(f'{split} at step {global_step}: {average}\n')
                    if split == "dev" and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {split} at step {global_step}: {average}\n')
                        save_names.append(f"{split}-best.ckpt")
                        
        if split in ["dev", "test"]:
            with open(Path(self.expdir) / f"{split}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{split}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

        return save_names
