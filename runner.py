# Author: Yuan Tseng
# Train/eval loop

# Modified from S3PRL 
# (Authors: Leo Yang, Andy T. Liu and S3PRL team, https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/runner.py)

import glob
import importlib
import math
import os
import random
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tensorboardX import SummaryWriter
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

import hub
from interfaces import Featurizer
from utils.helper import defaultdict, get_model_state, is_leader_process, show
from utils.optimizers import get_optimizer
from utils.schedulers import get_scheduler

SAMPLE_RATE = 16000


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner:
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = (
            torch.load(self.args.init_ckpt, map_location="cpu")
            if self.args.init_ckpt
            else {}
        )

        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream(
            self.upstream.model.preprocess if hasattr(self.upstream.model, "preprocess") else None ,self.upstream.model.preprocess_audio, self.upstream.model.preprocess_video
        )
        self.all_entries = [self.upstream, self.featurizer, self.downstream]

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f"[Runner] - Loading {name} weights from the previous experiment")
            model.load_state_dict(init_weight)

    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if (
            is_initialized()
            and trainable
            and any((p.requires_grad for p in model.parameters()))
        ):
            model = DDP(
                model, device_ids=[self.args.local_rank], find_unused_parameters=True
            )
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)

    def _get_upstream(self):
        Upstream = getattr(hub, self.args.upstream)
        ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt=ckpt_path,
            model_config=self.args.upstream_model_config,
            refresh=upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model=model,
            name="Upstream",
            trainable=self.args.upstream_trainable,
            interfaces=["preprocess_audio", "preprocess_video"],
        )

    def _get_featurizer(self):
        model = Featurizer(
            upstream=self.upstream.model,
            feature_selection=self.args.upstream_feature_selection,
            layer_selection=self.args.upstream_layer_selection,
            upstream_device=self.args.device,
            normalize=self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model=model,
            name="Featurizer",
            trainable=True,
            interfaces=["output_dim", "downsample_rate"],
        )

    def _get_downstream(self, preprocess, preprocess_audio, preprocess_video):
        expert = importlib.import_module(
            f"downstream_tasks.{self.args.downstream}.expert"
        )
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            preprocess=preprocess,
            preprocess_audio=preprocess_audio,
            preprocess_video=preprocess_video,
            upstream_dim=self.featurizer.model.output_dim,
            upstream_rate=self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args),
        ).to(self.args.device)

        return self._init_model(
            model=model,
            name="Downstream",
            trainable=True,
            interfaces=["get_dataloader", "log_records"],
        )

    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, self.config["runner"]["total_steps"], self.config["optimizer"]
        )
        self._load_weight(optimizer, "Optimizer")
        return optimizer

    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer, self.config["runner"]["total_steps"], self.config["scheduler"]
        )
        self._load_weight(scheduler, "Scheduler")
        return scheduler

    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                entry.model.train()
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get("scheduler"):
            scheduler = self._get_scheduler(optimizer)

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, "w")
        pbar = tqdm(
            total=self.config["runner"]["total_steps"],
            dynamic_ncols=True,
            desc="overall",
            file=tqdm_file,
        )
        init_step = self.init_ckpt.get("Step")
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get("Epoch", 0)
        train_split = self.config["runner"].get("train_dataloader", "train")
        while pbar.n < pbar.total:
            try:
                dataloader = self.downstream.model.get_dataloader(
                    train_split, epoch=epoch
                )
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(
                        dataloader.sampler, DistributedSampler
                    ):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            for batch_id, (wavs, frames, *others) in enumerate(
                tqdm(dataloader, dynamic_ncols=True, desc="train", file=tqdm_file)
            ):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    assert len(wavs) == len(frames)
                    if self.args.pooled_features_path and all(i == True for i in others[-1]):
                        source = None
                        features = dict()
                        # "wavs" is overloaded into saved features here
                        # can be list of Tensors, or list of list of Tensors
                        if isinstance(wavs[0], (list, tuple)):
                            features[self.args.upstream_feature_selection] = [torch.stack(layer).to(self.args.device) for layer in zip(*wavs)]
                        else:
                            features[self.args.upstream_feature_selection] = torch.stack(wavs).to(self.args.device)
                    else:
                        source = [
                            (
                                wav.float().to(self.args.device),
                                frame.float().to(self.args.device),
                            )
                            for wav, frame in zip(wavs, frames)
                        ]
                        if self.upstream.trainable:
                            features = self.upstream.model(source)
                        else:
                            with torch.no_grad():
                                features = self.upstream.model(source)
                        if self.args.pooled_features_path:
                            if batch_id == 0:
                                for feature_selection in features.keys():
                                    os.makedirs(f"{self.args.pooled_features_path}/{self.args.upstream}_{feature_selection}", exist_ok=True)

                            show(f"[Runner] - Save mean-pooled features of batch no. {batch_id}")
                            assert isinstance(others[-1][0], str)
                            with torch.no_grad():
                                for key, feature in features.items():
                                    if key[0] == '_':
                                        continue

                                    if isinstance(feature, (list, tuple)):
                                        feature = [layer.mean(dim=1, keepdim=True) for layer in feature]
                                    else:
                                        feature = feature.mean(dim=1, keepdim=True)

                                    for i, names_k in enumerate(others[-1]):
                                        if isinstance(feature, (list, tuple)):
                                            save_target = [f[i].detach().cpu() for f in feature]
                                        else:
                                            save_target = feature[i].detach().cpu()
                                        torch.save(save_target, f"{self.args.pooled_features_path}/{self.args.upstream}_{key}/{names_k}_pooled.pt")

                    features = self.featurizer.model(source, features)

                    loss = self.downstream.model(
                        train_split,
                        features,
                        *others,
                        records=records,
                    )
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config["runner"].get(
                        "gradient_accumulate_steps"
                    )
                    (loss / gradient_accumulate_steps).backward()
                    del loss

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[Runner] - CUDA out of memory at step {global_step}")
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config["runner"]["gradient_clipping"]
                )

                # optimize
                if math.isnan(grad_norm):
                    print(f"[Runner] - grad norm is NaN at step {global_step}")
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config["runner"]["log_step"] == 0:
                    self.downstream.model.log_records(
                        train_split,
                        records=records,
                        logger=logger,
                        global_step=global_step,
                        batch_ids=batch_ids,
                        total_batch_num=len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config["runner"]["eval_step"] == 0:
                    for split in self.config["runner"]["eval_dataloaders"]:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config["runner"]["save_step"] == 0:

                    def check_ckpt_num(directory):
                        max_keep = self.config["runner"]["max_keep"]
                        ckpt_pths = glob.glob(f"{directory}/states-*.ckpt")
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(
                                ckpt_pths,
                                key=lambda pth: int(pth.split("-")[-1].split(".")[0]),
                            )
                            for ckpt_pth in ckpt_pths[: len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)

                    check_ckpt_num(self.args.expdir)
                    save_names.append(f"states-{global_step}.ckpt")

                if len(save_names) > 0:
                    all_states = {
                        "Optimizer": optimizer.state_dict(),
                        "Step": global_step,
                        "Epoch": epoch,
                        "Args": self.args,
                        "Config": self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                    if scheduler:
                        all_states["Scheduler"] = scheduler.state_dict()

                    if is_initialized():
                        all_states["WorldSize"] = get_world_size()

                    save_paths = [
                        os.path.join(self.args.expdir, name) for name in save_names
                    ]
                    tqdm.write(f"[Runner] - Save the checkpoint to:")
                    for i, path in enumerate(save_paths):
                        tqdm.write(f"{i + 1}. {path}")
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1

        pbar.close()

        if is_leader_process():
            logger.close()

    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, frames, *others) in enumerate(
            tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)
        ):
            if batch_id > evaluate_steps:
                break

            assert len(wavs) == len(frames)
            if self.args.pooled_features_path and all(i == True for i in others[-1]):
                source = None
                features = dict()
                # "wavs" is overloaded into saved features here
                # can be list of Tensors, or list of list of Tensors
                if isinstance(wavs[0], (list, tuple)):
                    features[self.args.upstream_feature_selection] = [torch.stack(layer).to(self.args.device) for layer in zip(*wavs)]
                else:
                    features[self.args.upstream_feature_selection] = torch.stack(wavs).to(self.args.device)
            else:
                source = [
                    (
                        wav.float().to(self.args.device),
                        frame.float().to(self.args.device),
                    )
                    for wav, frame in zip(wavs, frames)
                ]
                with torch.no_grad():
                    features = self.upstream.model(source)
                if self.args.pooled_features_path:
                    show(f"[Runner] - Save mean-pooled features of batch no. {batch_id}")
                    assert isinstance(others[-1][0], str)
                    with torch.no_grad():
                        for key, feature in features.items():

                            if key[0] == '_':
                                continue

                            if isinstance(feature, (list, tuple)):
                                feature = [layer.mean(dim=1, keepdim=True) for layer in feature]
                            else:
                                feature = feature.mean(dim=1, keepdim=True)

                            for i, names_k in enumerate(others[-1]):
                                if isinstance(feature, (list, tuple)):
                                    save_target = [f[i].detach().cpu() for f in feature]
                                else:
                                    save_target = feature[i].detach().cpu()
                                torch.save(save_target, f"{self.args.pooled_features_path}/{self.args.upstream}_{key}/{names_k}_pooled.pt")


            with torch.no_grad():
                features = self.featurizer.model(source, features)
                self.downstream.model(
                    split,
                    features,
                    *others,
                    records=records,
                    batch_id=batch_id,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.model.log_records(
            split,
            records=records,
            logger=logger,
            global_step=global_step,
            batch_ids=batch_ids,
            total_batch_num=len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        raise NotImplementedError("not updated to audio-visual models")
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])
