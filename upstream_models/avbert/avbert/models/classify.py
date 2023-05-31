import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_helper import ResNetBasicHead as VisualHead
from .audio_head_helper import ResNetBasicHead as AudioHead
from .video_model_builder import _POOL1
from utils.weight_init_helper import init_weights
from .build import MODEL_REGISTRY
from .avbert import AVBert


def get_visual_dim_in(
    trans_func,
    width_per_group,
    pooling,
):
    """
    Compute the input dimension to the VisualClassifyHead.
    args:
        trans_func (string): transform function to be used to contrusct each
            ResBlock.
        width_per_group (int): width of each group.
        pooling (bool): Whether to use the output of the Transformer or
            the ConvNet.
    returns:
        dim_in (int or list): the input dimension.
    """
    if trans_func == "basic_transform":
        factor = 2 ** 3
    elif trans_func == "bottleneck_transform":
        factor = 4 * (2 ** 3)
    else:
        raise NotImplementedError(
            "Does not support {} transfomration".format(trans_func)
        )
    dim_in = [width_per_group * factor]

    if pooling:
        dim_in = sum(dim_in)

    return dim_in


def get_audio_dim_in(
    trans_func,
    width_per_group,
):
    """
    Compute the input dimension to the AudioClassifyHead.
    args:
        trans_func (string): transform function to be used to contrusct each
            ResBlock.
        width_per_group (int): width of each group.
    returns:
        dim_in (int): the input dimension.
    """
    if trans_func == "basic_transform":
        factor = 2 ** 3
    elif trans_func == "bottleneck_transform":
        factor = 4 * (2 ** 3)
    else:
        raise NotImplementedError(
            "Does not support {} transfomration".format(trans_func)
        )
    dim_in = width_per_group * factor
    return dim_in


def get_visual_pool_size(
    vis_arch,
    num_frames,
    crop_size,
):
    """
    Compute the pooling size used in VisualClassifyHead.
    args:
        vis_arch (string): the architecture of the visual conv net.
        num_frames (int): number of frames per clip.
        crop_size (int): spatial size of frames.
    returns:
        pool_size (list): list of p the kernel sizes of spatial tempoeral
            poolings, temporal pool kernel size, height pool kernel size,
            width pool kernel size in order.
    """
    _pool_size = _POOL1[vis_arch]
    _num_frames = num_frames // 2
    pool_size = [
        [
            _num_frames // _pool_size[0][0],
            math.ceil(crop_size / 32) // _pool_size[0][1],
            math.ceil(crop_size / 32) // _pool_size[0][2],
        ]
    ]

    return pool_size


def get_audio_pool_size(
    audio_frequency,
    audio_time,
):
    """
    Compute the pooling size used in AudioClassifyHead.
    args:
        audio_frequency (int): frequency dimension of the audio clip.
        audio_time (int): time dimension of the audio clip.
    returns:
        pool_size (list): list of the kernel sizes of an avg pooling,
            frequency pool kernel size, time pool kernel size in order.
    """
    pool_size = [
        math.ceil(audio_frequency / 16),
        math.ceil(audio_time / 16),
    ]

    return pool_size


class ConcatHead(nn.Module):
    """
    Concatenation head for the conv net outputs.
    This layer takes as input the concateneation of the outputs of audio and
    visual convolutional nets and performs a fully-connected projection.
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            num_classes (int): the channel dimension of the output.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ConcatHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
        return x


class ConcatSeqHead(nn.Module):
    """
    Concatenation head for the multimodal Transformer outputs.
    This layer takes as input the Transformer outputs from audio and visual
    sequences, and performs a fully-connected projection.
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        num_classes,
        dropout_rate=0.0,
        eps=1e-5,
        act_func="softmax",
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            dim_hidden (int): the channel dimension of the intermediate output.
            num_classes (int): the channel dimension of the output.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            eps (float): epsilon for normalization.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ConcatSeqHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.ln = nn.LayerNorm(dim_hidden, eps=eps)
        self.projection = nn.Linear(dim_hidden, num_classes)

        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, y):
        if hasattr(self, "dropout1"):
            x = self.dropout1(x)
            y = self.dropout2(y)
        x = self.fc1(x)
        y = self.fc2(y)
        out = self.ln(x + y)
        out = self.projection(out)
        if not self.training:
            out = self.act(out)
        return out


class ClassifyHead(nn.Module):
    """
    Classification head.
    For linear evaluation, only this classification head will be trained.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ClassifyHead, self).__init__()
        self.cfg = cfg
        if cfg.MODEL.TASK in ["VisualClassify"]:
            visual_dim_in = get_visual_dim_in(
                cfg.RESNET.TRANS_FUNC,
                cfg.RESNET.WIDTH_PER_GROUP,
                False,
            )
            visual_pool_size = get_visual_pool_size(
                cfg.VIS.ARCH,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.CROP_SIZE,
            )
            self.visual_head = VisualHead(
                dim_in=visual_dim_in,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=visual_pool_size,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                fusion=cfg.MODEL.DOWNSTREAM_FUSION,
            )
        elif cfg.MODEL.TASK in ["AudioClassify"]:
            audio_dim_in = get_audio_dim_in(
                cfg.AUDIO_RESNET.TRANS_FUNC,
                cfg.AUDIO_RESNET.WIDTH_PER_GROUP,
            )
            audio_pool_size = get_audio_pool_size(
                cfg.DATA.AUDIO_FREQUENCY,
                cfg.DATA.AUDIO_TIME,
            )
            self.audio_head = AudioHead(
                dim_in=audio_dim_in,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=audio_pool_size,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                fusion=cfg.MODEL.DOWNSTREAM_FUSION,
            )
        elif cfg.MODEL.TASK in ["MultimodalSequenceClassify"]:
            visual_pool_size = get_visual_pool_size(
                cfg.VIS.ARCH,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.CROP_SIZE,
            )
            audio_dim_in = get_audio_dim_in(
                cfg.AUDIO_RESNET.TRANS_FUNC,
                cfg.AUDIO_RESNET.WIDTH_PER_GROUP,
            )
            audio_pool_size = get_audio_pool_size(
                cfg.DATA.AUDIO_FREQUENCY,
                cfg.DATA.AUDIO_TIME,
            )
            visual_dim_in = get_visual_dim_in(
                cfg.RESNET.TRANS_FUNC,
                cfg.RESNET.WIDTH_PER_GROUP,
                True,
            )
            self.concat_head = ConcatHead(
                visual_dim_in + audio_dim_in,
                cfg.MODEL.NUM_CLASSES,
                cfg.MODEL.DROPOUT_RATE,
                cfg.MODEL.HEAD_ACT,
            )
            self.concat_seq_head = ConcatSeqHead(
                2 * cfg.TRANSFORMER.HIDDEN_SIZE,
                cfg.TRANSFORMER.HIDDEN_SIZE,
                cfg.MODEL.NUM_CLASSES,
                cfg.MODEL.DROPOUT_RATE,
                cfg.MODEL.EPSILON,
                cfg.MODEL.HEAD_ACT,
            )

        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def visual_forward(self, visual_feature_map):
        return [self.visual_head(visual_feature_map)]

    def audio_forward(self, audio_feature_map):
        return [self.audio_head(audio_feature_map)]

    def multimodal_seq_forward(self, v_f, a_f, m_t):
        batch_size = m_t.shape[0]
        seqlen = (m_t.shape[1] - 2) // 2
        mv = torch.cat(
            (m_t[:, 0].unsqueeze(1).expand(-1, seqlen, -1), m_t[:, 1:1+seqlen]),
            dim=-1,
        )
        ma = torch.cat(
            (m_t[:, 1+seqlen].unsqueeze(1).expand(-1, seqlen, -1), m_t[:, 2+seqlen:]),
            dim=-1,
        )
        mv = mv.view(batch_size * seqlen, -1)
        ma = ma.view(batch_size * seqlen, -1)
        visual_logit, audio_logit = v_f, a_f
        clip_logit = self.concat_head(
            torch.cat((visual_logit, audio_logit), dim=-1)
        )
        seq_logit = self.concat_seq_head(mv, ma)
        multimodal_logit = [clip_logit, seq_logit]

        return multimodal_logit

    def forward(self, feature_maps):
        if self.cfg.MODEL.TASK in ["VisualClassify"]:
            return self.visual_forward(*feature_maps)
        elif self.cfg.MODEL.TASK in ["AudioClassify"]:
            return self.audio_forward(*feature_maps)
        elif self.cfg.MODEL.TASK in ["MultimodalSequenceClassify"]:
            return self.multimodal_seq_forward(*feature_maps)


@MODEL_REGISTRY.register()
class VisualClassify(nn.Module):
    """
    Visual classifier
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VisualClassify, self).__init__()
        self.cfg = cfg
        self.visual_conv = MODEL_REGISTRY.get(cfg.VIS.MODEL_NAME)(cfg)
        assert cfg.MODEL.DOWNSTREAM_FUSION == "late", \
            f"Visual Classifier needs late fusion"
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def forward(self, visual_clip, protocol):
        if protocol == "linear_eval":
            return (self.visual_conv.get_feature_map(visual_clip), )
        else:
            return [self.visual_conv(visual_clip)]


@MODEL_REGISTRY.register()
class AudioClassify(nn.Module):
    """
    Audio classifier
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AudioClassify, self).__init__()
        self.cfg = cfg
        self.audio_conv = MODEL_REGISTRY.get(cfg.AUD.MODEL_NAME)(cfg)
        assert cfg.MODEL.DOWNSTREAM_FUSION == "late", \
            f"Audio Classifier needs late fusion"
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def forward(self, audio_clip, protocol):
        if protocol == "linear_eval":
            return (self.audio_conv.get_feature_map(audio_clip), )
        else:
            return [self.audio_conv(audio_clip)]


@MODEL_REGISTRY.register()
class MultimodalSequenceClassify(nn.Module):
    """
    Multimodal sequence classifier
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(MultimodalSequenceClassify, self).__init__()
        self.cfg = cfg
        self.avbert = AVBert(cfg)
        if self.cfg.SOLVER.PROTOCOL == "finetune":
            visual_pool_size = get_visual_pool_size(
                cfg.VIS.ARCH,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.CROP_SIZE,
            )
            audio_dim_in = get_audio_dim_in(
                cfg.AUDIO_RESNET.TRANS_FUNC,
                cfg.AUDIO_RESNET.WIDTH_PER_GROUP,
            )
            audio_pool_size = get_audio_pool_size(
                cfg.DATA.AUDIO_FREQUENCY,
                cfg.DATA.AUDIO_TIME,
            )
            visual_dim_in = get_visual_dim_in(
                cfg.RESNET.TRANS_FUNC,
                cfg.RESNET.WIDTH_PER_GROUP,
                True,
            )
            self.concat_head = ConcatHead(
                visual_dim_in + audio_dim_in,
                cfg.MODEL.NUM_CLASSES,
                cfg.MODEL.DROPOUT_RATE,
                cfg.MODEL.HEAD_ACT,
            )
            self.concat_seq_head = ConcatSeqHead(
                2 * cfg.TRANSFORMER.HIDDEN_SIZE,
                cfg.TRANSFORMER.HIDDEN_SIZE,
                cfg.MODEL.NUM_CLASSES,
                cfg.MODEL.DROPOUT_RATE,
                cfg.MODEL.EPSILON,
                cfg.MODEL.HEAD_ACT,
            )

        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def multimodal_seq_forward(self, v_f, a_f, m_t):
        batch_size = m_t.shape[0]
        seqlen = (m_t.shape[1] - 2) // 2
        mv = torch.cat(
            (m_t[:, 0].unsqueeze(1).expand(-1, seqlen, -1), m_t[:, 1:1+seqlen]),
            dim=-1,
        )
        ma = torch.cat(
            (m_t[:, 1+seqlen].unsqueeze(1).expand(-1, seqlen, -1), m_t[:, 2+seqlen:]),
            dim=-1,
        )
        mv = mv.view(batch_size * seqlen, -1)
        ma = ma.view(batch_size * seqlen, -1)
        visual_logit, audio_logit = v_f, a_f
        clip_logit = self.concat_head(
            torch.cat((visual_logit, audio_logit), dim=-1)
        )
        seq_logit = self.concat_seq_head(mv, ma)
        multimodal_logit = [clip_logit, seq_logit]

        return multimodal_logit

    def forward(self, visual_seq, audio_seq, protocol):
        conv_outputs, single_outputs, multi_output = self.avbert(
            visual_seq=visual_seq, audio_seq=audio_seq
        )
        feature_maps = (conv_outputs[0], conv_outputs[1], multi_output)
        feature_maps = (
            self.avbert.visual_conv.head(feature_maps[0]),
            self.avbert.audio_conv.head(feature_maps[1]),
            feature_maps[2],
        )
        if protocol == "linear_eval":
            return feature_maps
        else:
            return self.multimodal_seq_forward(*feature_maps)
