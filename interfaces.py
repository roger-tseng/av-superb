import random
import sys
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.helper import show

TOLERABLE_SEQLEN_DIFF = 5
AUDIO_SAMPLE_RATE = 16000
VIDEO_SAMPLE_RATE = 16
HEIGHT = 112
WIDTH = 112
MIN_SEC = 2
MAX_SEC = 4


class Hook:
    def __init__(self, module_path, transform, unique_identifier=None):
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)


class initHook(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for hook in instance.hooks:
            if hook.handler is None:
                instance._register_hook_handler(hook)
        return instance


class UpstreamBase(nn.Module, metaclass=initHook):
    def __init__(
        self,
        hooks: List[Tuple] = None,
        hook_postprocess: Callable[
            [List[Tuple[str, Tensor]]], List[Tuple[str, Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Args:
            hooks: each Tuple is an argument list for the Hook initializer
        """
        super().__init__()
        self.hooks: List[Hook] = [Hook(*hook) for hook in hooks] if hooks else []
        self.hook_postprocess = hook_postprocess
        self._hook_hiddens: List[Tuple(str, Tensor)] = []

    def remove_all_hooks(self):
        for hook in self.hooks:
            hook.handler.remove()
        self.hooks.clear()

    def remove_hook(self, unique_identifier: str):
        updated_hooks = []
        for hook in self.hooks:
            if hook.unique_identifier == unique_identifier:
                hook.handler.remove()
            else:
                updated_hooks.append(hook)
        self.hooks = updated_hooks

    def add_hook(self, *args, **kwargs):
        hook = Hook(*args, **kwargs)
        self._register_hook_handler(hook)
        self.hooks.append(hook)

    def _register_hook_handler(self, hook: Hook):
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):
            show(
                f"[UpstreamBase] - {hook.module_path} is not a valid nn.Module. Skip.",
                file=sys.stderr,
            )
            return

        if callable(hook.handler):
            show(
                f"[UpstreamBase] - Existing hook handler for {hook.unique_identifier} is found. Remove the existing one.",
                file=sys.stderr,
            )
            hook.handler.remove()

        def generate_hook_handler(hiddens: List, hook: Hook):
            def hook_handler(self, input, output):
                hiddens.append((hook.unique_identifier, hook.transform(input, output)))

            return hook_handler

        hook.handler = module.register_forward_hook(
            generate_hook_handler(self._hook_hiddens, hook)
        )

    def __call__(self, wavs: List[Tensor], *args, **kwargs):
        self._hook_hiddens.clear()

        result = super().__call__(wavs, *args, **kwargs) or {}
        # paths = [pth if isinstance(pth, str) else pth[0] for _, _, pth in wavs]
        assert isinstance(result, dict)

        if len(self._hook_hiddens) > 0:
            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            names, hiddens = zip(*hook_hiddens)
            if "fusion" in names[0]:
                key = "fusion_feats"
            elif "video" in names[0]:
                key = "video_feats"
            elif "audio" in names[0]:
                key = "audio_feats"

            if (
                result.get("_hidden_states_info") is not None
                or result.get(key) is not None
            ):
                show(
                    f"[UpstreamBase] - If there are registered hooks, '_hidden_states_info' and '{key}' "
                    "are reserved and should not be included in child class's return dict.",
                    file=sys.stderr,
                )
                raise ValueError

            # if not isinstance(wavs[-1][-1], tuple):
            #     for i in range(len(paths)):
            #         torch.save(
            #             [hidden[i].cpu() for hidden in hiddens], paths[i] + "_fusion"
            #         )

            result["_hidden_states_info"], result[key] = names, hiddens

        return result


class Featurizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = "Featurizer"

        upstream.eval()
        length = random.randint(MIN_SEC, MAX_SEC)
        audio_samples = length * AUDIO_SAMPLE_RATE
        video_samples = length * VIDEO_SAMPLE_RATE
        paired_wavs = [
            (
                torch.randn(audio_samples),
                torch.ones(video_samples, 3, HEIGHT, WIDTH, dtype=torch.uint8),
            )
            for i in range(2)
        ]

        with torch.no_grad():
            if hasattr(upstream, "preprocess"):

                paired = [
                    (
                        upstream.preprocess(video, audio, VIDEO_SAMPLE_RATE, AUDIO_SAMPLE_RATE)
                    )
                    for audio, video in paired_wavs
                ]
                paired_input = [
                    (
                        audio.to(upstream_device),
                        video.to(upstream_device),
                    )
                    for video, audio in paired
                ]
                paired_features = upstream(paired_input)
            elif hasattr(upstream, "preprocess_audio") and hasattr(
                upstream, "preprocess_video"
            ):
                paired_input = [
                    (
                        upstream.preprocess_audio(audio, AUDIO_SAMPLE_RATE).to(
                            upstream_device
                        ),
                        upstream.preprocess_video(video, VIDEO_SAMPLE_RATE).to(
                            upstream_device
                        ),
                    )
                    for audio, video in paired_wavs
                ]
                paired_features = upstream(paired_input)
            else:
                show(
                    f"[{self.name}] - Error: Your upstream model does not implement its preprocessing functions."
                    f" Please follow upstream_models/example/expert.py to add your preprocess_audio and preprocess_video functions for your upstream.",
                    file=sys.stderr,
                )
                raise NotImplementedError
        if feature_selection not in paired_features:
            if "hidden_states" in paired_features:
                show(
                    f"[{self.name}] - Warning: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' Using "hidden_states" as the default key.',
                    file=sys.stderr,
                )
                feature_selection = "hidden_states"
            else:
                show(
                    f"[{self.name}] - Error: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' The default key "hidden_states" is also not supported.'
                    f" Please specify -s with the following options: {list(paired_wavs.keys())}",
                    file=sys.stderr,
                )
                raise ValueError
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection
        self.normalize = normalize

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them.",
                file=sys.stderr,
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature = self._weighted_sum([f.cpu() for f in feature])
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        if hasattr(upstream, "get_downsample_rates"):
            self.downsample_rate = upstream.get_downsample_rates(feature_selection)
            show(
                f"[{self.name}] - The selected feature {feature_selection}'s downsample rate is {self.downsample_rate}",
                file=sys.stderr,
            )
        else:
            self.downsample_rate = round(
                max(len(wav[0]) for wav in paired_wavs) / feature.size(1)
            )
            # TODO: add back in downsample rates for padding
            # show(
            #     f"[{self.name}] - Warning: The provided upstream does not give statis downsample rate"
            #     ' by the "get_downsample_rates" interface (see upstream/example/expert.py).'
            #     " The downsample rate is calculated dynamically basing on the shape of the"
            #     f" input waveforms v.s. the output features: {self.downsample_rate}",
            #     file=sys.stderr,
            # )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]

        if isinstance(feature, (list, tuple)) and isinstance(self.layer_selection, int):
            feature = feature[self.layer_selection]

        return feature

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), (
            "If you run into this error, there is a great chance"
            " you are finetuning the upstream with wav2vec2's transformer blocks"
            " in weighted-sum mode (default), including wav2vec2, hubert, and decoar2."
            " These models use the layerdrop technique which causes the different number"
            " of layer forwards between different model forwards, resulting in different"
            " number of hidden states for different model forwards. Hence, finetuning"
            " these upstreams is essentially incompatible with weight-sum mode unless"
            " you turn off the layerdrop option in fairseq. See:"
            " https://github.com/pytorch/fairseq/blob/f6abcc2a67328bee8b15c596bb626ce2d720aae6/fairseq/models/wav2vec/wav2vec2.py#L857"
            " However, since finetuning upstreams will backward the gradient through all layers"
            " which serves the same functionality as weighted-sum: all layers can be used for different"
            " downstream tasks. Hence instead of finetuning upstream with weighted-sum, we suggest to"
            " follow the more common setting: finetuning upstream with the last layer. Please use the"
            " following options: --upstream_trainable --upstream_feature_selection last_hidden_state."
            " Or: -f -s last_hidden_state"
        )
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def tolist(self, paired_wavs: List[Tuple[Tensor, Tensor]], paired_feature: Tensor, lens=None):
        assert paired_feature.dim() == 3, "(batch_size, max_seq_len, feat_dim)"
        # TODO: check if this is needed
        # feature_len = [round(len(wav[0]) / self.downsample_rate) for wav in paired_wavs]
        # length_diff = abs(
        #     paired_feature.size(1)
        #     - round(max([len(wav[0]) for wav in paired_wavs]) / self.downsample_rate)
        # )
        # assert (
        #     length_diff < TOLERABLE_SEQLEN_DIFF
        # ), f"{length_diff} >= {TOLERABLE_SEQLEN_DIFF}, {paired_feature.size(1)}, {max([len(wav[0]) for wav in paired_wavs])}"
        # feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
        if lens is not None:
            assert len(lens) == len(paired_feature)
            feature = [f[:l] for f, l in zip(paired_feature, lens)]
        feature = [f for f in paired_feature]
        return feature

    def forward(
        self,
        paired_wavs: List[Tuple[Tensor, Tensor]],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
        lens=None, 
    ):
        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            feature = self._weighted_sum(feature)

        return self.tolist(paired_wavs, feature, lens)
