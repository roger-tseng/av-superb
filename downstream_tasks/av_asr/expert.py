"""
Custom class for training/testing code for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/expert.py
"""
import math
import os
import random

import editdistance
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .dataset import RandomDataset
from .fairseq_dictionary import Dictionary
from .model import Model


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

        self.dictionary = Dictionary.load("downstream_tasks/av_asr/char.dict")

        self.train_dataset = RandomDataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            split="train", 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'], 
            **self.datarc
        )
        self.dev_dataset = RandomDataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'], 
            split="val", 
            **self.datarc
        )
        self.test_dataset = RandomDataset(
            preprocess, 
            preprocess_audio, 
            preprocess_video, 
            upstream=kwargs['upstream'],
            pooled_features_path=kwargs['pooled_features_path'],
            upstream_feature_selection=kwargs['upstream_feature_selection'], 
            split="test", 
            **self.datarc
        )

        self.connector = nn.Linear(upstream_dim, self.modelrc["input_dim"])
        self.model = Model(
            output_class_num=self.train_dataset.class_num, **self.modelrc
        )
        self.blank = self.dictionary.bos()
        self.objective = nn.CTCLoss(blank=self.blank, zero_infinity=True)
        self.register_buffer(
            "best_score", torch.ones(1) * 1000
        )  # increased from 100 since WER can be > 100, and we want at least one checkpoint to get saved

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

    def _compute_metrics(
        self, pred_tokens_all, pred_words_all, target_tokens_all, target_words_all
    ):
        unit_error_sum = 0.0
        word_error_sum = 0.0
        unit_length_sum = 0
        word_length_sum = 0

        for pred_tokens, pred_words, target_tokens, target_words in zip(
            pred_tokens_all, pred_words_all, target_tokens_all, target_words_all
        ):
            pred_tokens = pred_tokens.split()
            target_tokens = target_tokens.split()
            unit_error_sum += editdistance.eval(pred_tokens, target_tokens)
            unit_length_sum += len(target_tokens)
            word_error_sum += editdistance.eval(pred_words, target_words)
            word_length_sum += len(target_words)

        uer, wer = 100.0, 100.0
        if unit_length_sum > 0:
            uer = 100.0 * unit_error_sum / unit_length_sum
        else:
            print("Warning: Unit Length Sum was zero!")
        if word_length_sum > 0:
            wer = 100.0 * word_error_sum / word_length_sum
        else:
            print("Warning: Word Length Sum was zero!")
        if unit_length_sum == 0 and word_length_sum == 0:
            print("Terminating on zero units found. Inputs were:")
            print("\tpred_tokens_all", pred_tokens_all)
            print("\tpred_words_all", pred_words_all)
            print("\ttarget_tokens_all", target_tokens_all)
            print("\ttarget_words_all", target_words_all)
        return uer, wer

    def _decode(self, log_probs, input_lens):
        pred_tokens_batch = []
        pred_words_batch = []
        for log_prob, in_len in zip(log_probs, input_lens):
            log_prob = log_prob[:in_len].unsqueeze(0)
            pred_token_ids = log_prob.argmax(dim=-1).unique_consecutive()
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()
            pred_tokens = self.dictionary.string(pred_token_ids)
            pred_words = pred_tokens.replace(" ", "").replace("|", " ").strip().split()
            pred_tokens_batch.append(pred_tokens)
            pred_words_batch.append(pred_words)
        return pred_tokens_batch, pred_words_batch

    def _get_log_probs(self, features):
        features, features_len = self._get_lens_and_pad(features)
        features = self.connector(features)
        logits, log_probs_len = self.model(features, features_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs, log_probs_len

    def _get_lens_and_pad(self, seq_list, device=None):
        if device is None:
            device = seq_list[0].device
        seq_lens = torch.IntTensor([len(seq) for seq in seq_list])
        all_seqs = pad_sequence(seq_list, batch_first=True).to(device=device)
        return all_seqs, seq_lens

    # Interface
    def forward(self, split, features, labels, basenames, records, **kwargs):
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
        device = features[0].device

        labels, labels_len = self._get_lens_and_pad(labels, device)

        try:
            unpadded_features = features
            # unpadded_features = []
            # for i in range(len(features)):
            #     unpadded_features.append(features[i][: lens[i]])
            log_probs, log_probs_len = self._get_log_probs(unpadded_features)
        except:
            print("Failed to pad features of shape", features.shape)

        loss = self.objective(
            log_probs.transpose(0, 1), labels, log_probs_len, labels_len
        )
        """
        if loss == 0.0:
            print('Got loss of 0.0!')
            print('labels was len', len(labels))
            print('log_probs was shape', log_probs.shape)
            print('passed log_probs_len', log_probs_len)
            print('and labels_len', labels_len)
            print('for video with len', lens)
            # exit(0)
        """
        records["loss"].append(loss.item())

        target_tokens_batch = []
        target_words_batch = []
        for label in labels:
            label_idx = (label != self.dictionary.pad()) & (
                label != self.dictionary.eos()
            )
            target_token_ids = label[label_idx].tolist()
            target_tokens = self.dictionary.string(target_token_ids)
            target_words = (
                target_tokens.replace(" ", "").replace("|", " ").strip().split()
            )
            target_tokens_batch.append(target_tokens)
            target_words_batch.append(target_words)

        with torch.no_grad():
            pred_tokens_batch, pred_words_batch = self._decode(
                log_probs.float().contiguous().cpu(), log_probs_len
            )

        records["target_tokens"] += target_tokens_batch
        records["target_words"] += target_words_batch
        records["pred_tokens"] += pred_tokens_batch
        records["pred_words"] += pred_words_batch

        return loss

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

        loss = torch.FloatTensor(records["loss"]).mean().item()
        print(f"{split} loss: {loss}")

        uer, wer = self._compute_metrics(
            records["pred_tokens"],
            records["pred_words"],
            records["target_tokens"],
            records["target_words"],
        )
        logger.add_scalar(f"asr/{split}-loss", loss, global_step=global_step)
        logger.add_scalar(f"asr/{split}-uer", uer, global_step=global_step)
        logger.add_scalar(f"asr/{split}-wer", wer, global_step=global_step)
        print(f"{split} uer: {uer}")
        print(f"{split} wer: {wer}")

        save_names = []
        if split == "dev" and uer < self.best_score:
            self.best_score = torch.ones(1) * uer
            save_names.append(f"{split}-best.ckpt")
        return save_names
