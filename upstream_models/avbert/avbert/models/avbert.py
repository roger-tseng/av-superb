import copy
import numpy as np

import torch
import torch.nn as nn

from .build import MODEL_REGISTRY
from .encoding_helper import SummaryEncoding
from .bert import (
    BertConfig,
    BertEmbeddings,
    Bert
)
from .albert import (
    AlbertConfig,
    AlbertEmbeddings,
    Albert
)


def _identity(x, attention_mask=None, modality_idx=0):
    return [x]


@MODEL_REGISTRY.register()
class AVBert(nn.Module):
    """
    Multimodal Transformer model builder.
    It builds a multimodal Transformer backbone that takes audio and visual
    sequences as the input.
    """
    def __init__(self, cfg):
        super(AVBert, self).__init__()
        self.cfg = cfg
        # Shring across layers
        transformer_cfg_func = \
            AlbertConfig if cfg.TRANSFORMER.SHARING_ACROSS_LAYERS else BertConfig
        transformer_embeddings = \
            AlbertEmbeddings if cfg.TRANSFORMER.SHARING_ACROSS_LAYERS else \
            BertEmbeddings
        transformer_model = \
            Albert if cfg.TRANSFORMER.SHARING_ACROSS_LAYERS else Bert
        transformer_cfg = transformer_cfg_func.from_dict(
            {key.lower(): value for key, value in cfg.TRANSFORMER.items()}
        )

        self.num_modalities = 2 if 'visual_audio' in cfg.MODEL.ARCH else 1
        self.idx2modality = []
        if 'visual' in cfg.MODEL.ARCH:
            self.visual_conv = MODEL_REGISTRY.get(cfg.VIS.MODEL_NAME)(cfg)
            self.visual_fc = nn.Linear(
                self.visual_conv.head.output_size, transformer_cfg.hidden_size
            )
            self.visual_norm = nn.LayerNorm(
                transformer_cfg.hidden_size, transformer_cfg.layer_norm_eps
            )
            self.idx2modality.append('visual')
        if 'audio' in cfg.MODEL.ARCH:
            self.audio_conv = MODEL_REGISTRY.get(cfg.AUD.MODEL_NAME)(cfg)
            self.audio_fc = nn.Linear(
                self.audio_conv.output_size, transformer_cfg.hidden_size
            )
            self.audio_norm = nn.LayerNorm(
                transformer_cfg.hidden_size, transformer_cfg.layer_norm_eps
            )
            self.idx2modality.append('audio')

        self.modality2idx = \
            {value: idx for idx, value in enumerate(self.idx2modality)}

        # BOS embeddings
        self.summary_encoding = SummaryEncoding(
            self.idx2modality,
            transformer_cfg.hidden_size,
            transformer_cfg.use_mean_pooling,
            cfg.DATA.SEQUENCE_LENGTH,
            transformer_cfg.layer_norm_eps
        )

        # Modality fusion strategies
        if cfg.MODEL.FUSION == 'early':
            assert 'visual_audio' in cfg.MODEL.ARCH
            for i in range(self.num_modalities):
                setattr(
                    self,
                    f"single_{self.idx2modality[i]}_embeddings",
                    lambda x: x
                )
                setattr(
                    self,
                    f"single_{self.idx2modality[i]}_transformer",
                    _identity
                )
            transformer_cfg.num_modality_groups = 1
            self.multi_embeddings = \
                transformer_embeddings(transformer_cfg, True)
            self.multi_transformer = transformer_model(transformer_cfg)
        elif cfg.MODEL.FUSION == 'mid':
            assert 'visual_audio' in cfg.MODEL.ARCH
            transformer_cfg.num_modality_groups = self.num_modalities + 1
            for i in range(self.num_modalities):
                self.add_module(
                    f"single_{self.idx2modality[i]}_embeddings",
                    transformer_embeddings(transformer_cfg, False)
                )
            self.multi_embeddings = \
                transformer_embeddings(transformer_cfg, True)
            # Sharing across modalities
            if cfg.TRANSFORMER.SHARING_ACROSS_MODELS:
                self.transformer = transformer_model(transformer_cfg)
                for i in range(self.num_modalities):
                    setattr(
                        self,
                        f"single_{self.idx2modality[i]}_transformer",
                        self.transformer
                    )
                self.multi_transformer = self.transformer
            else:
                for i in range(self.num_modalities):
                    self.add_module(
                        f"single_{self.idx2modality[i]}_transformer",
                        transformer_model(transformer_cfg)
                    )
                self.multi_transformer = \
                    transformer_model(transformer_cfg)
        elif cfg.MODEL.FUSION == 'late':
            transformer_cfg.num_modality_groups = self.num_modalities
            for i in range(self.num_modalities):
                self.add_module(
                    f"single_{self.idx2modality[i]}_embeddings",
                    transformer_embeddings(transformer_cfg, False)
                )
            # Sharing across modalities
            if cfg.TRANSFORMER.SHARING_ACROSS_MODELS:
                self.transformer = transformer_model(transformer_cfg)
                for i in range(self.num_modalities):
                    setattr(
                        self,
                        f"single_{self.idx2modality[i]}_transformer",
                        self.transformer
                    )
            else:
                for i in range(self.num_modalities):
                    self.add_module(
                        f"single_{self.idx2modality[i]}_transformer",
                        transformer_model(transformer_cfg)
                    )
            self.multi_embeddings = None
            self.multi_transformer = None
        else:
            raise NotImplementedError(
                "Does not support {} fusion".format(cfg.MODEL.FUSION)
            )

    def forward(
        self,
        visual_seq=None,
        audio_seq=None,
    ):
        batch_size, seqlen = \
            visual_seq[0].size()[:2] if visual_seq is not None else \
            audio_seq.size()[:2]

        # ConvNet
        conv_outputs = []
        _conv_outputs = []
        idx2modality = []
        if 'visual' in self.cfg.MODEL.ARCH and visual_seq is not None:
            nchannels, _, H, W = visual_seq[0].size()[2:]
            vconv_repr = self.visual_conv.get_feature_map(
                [
                    t.view(
                        batch_size * seqlen,
                        nchannels,
                        -1,
                        H,
                        W,
                    )
                    for t in visual_seq
                ]
            )
            conv_outputs.append(vconv_repr)
            _conv_outputs.append(
                self.visual_conv.get_logit(vconv_repr).view(batch_size, seqlen, -1)
            )
            idx2modality.append('visual')
        if 'audio' in self.cfg.MODEL.ARCH and audio_seq is not None:
            nchannels, frequency, time = audio_seq.size()[2:]
            aconv_repr = self.audio_conv.get_feature_map(
                audio_seq.view(
                    batch_size * seqlen,
                    nchannels,
                    frequency,
                    time,
                )
            )
            conv_outputs.append(aconv_repr)
            _conv_outputs.append(
                self.audio_conv.get_logit(aconv_repr).view(batch_size, seqlen, -1)
            )
            idx2modality.append('audio')
        # assert len({len(_conv_outputs), self.num_modalities}) == 1
        modality2idx = {value: idx for idx, value in enumerate(idx2modality)}
        num_modalities = len(idx2modality)

        # Transformers
        # Single modality
        # Projection and normalization
        # Prepend summary embeddings
        single_inputs = []
        for idx in range(num_modalities):
            fc = getattr(self, f"{idx2modality[idx]}_fc")
            norm = getattr(self, f"{idx2modality[idx]}_norm")
            conv_repr_prj = norm(fc(_conv_outputs[idx]))
            _idx = self.modality2idx[idx2modality[idx]]
            single_inputs.append(
                self.summary_encoding(conv_repr_prj, _idx)
            )

        att_mask = torch.ones(
            batch_size, 1 + seqlen,
            dtype=torch.long,
            device=_conv_outputs[0].device,
        )

        single_outputs = []
        for idx in range(num_modalities):
            s_embeddings = getattr(
                self,
                f"single_{idx2modality[idx]}_embeddings"
            )
            s_transformer = getattr(
                self,
                f"single_{idx2modality[idx]}_transformer"
            )
            _idx = self.modality2idx[idx2modality[idx]]
            single_output = s_transformer(
                s_embeddings(single_inputs[idx]),
                attention_mask=att_mask,
                modality_idx=_idx
            )[0]
            single_outputs.append(single_output)


        # Multi modality
        multi_output = None
        if self.multi_transformer is not None and num_modalities == 2:
            multi_input = torch.cat(single_outputs, dim=1)
            token_type_ids = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        1 + seqlen,
                        dtype=torch.long,
                        device=multi_input.device
                    ),
                    torch.ones(
                        batch_size,
                        1 + seqlen,
                        dtype=torch.long,
                        device=multi_input.device
                    )
                ],
                dim=1
            )
            input_shape = multi_input.size()[:-1]
            position_ids = torch.cat(
                [
                    torch.arange(
                        1 + seqlen,
                        dtype=torch.long,
                        device=multi_input.device
                    ),
                    torch.arange(
                        1 + seqlen,
                        dtype=torch.long,
                        device=multi_input.device
                    )
                ]
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

            multi_attention_mask = torch.cat(
                [att_mask, att_mask],
                dim=1,
            )

            multi_output = self.multi_transformer(
                self.multi_embeddings(
                    multi_input,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids
                ),
                attention_mask=multi_attention_mask,
                modality_idx=self.num_modalities,
            )[0]

        return conv_outputs, single_outputs, multi_output
