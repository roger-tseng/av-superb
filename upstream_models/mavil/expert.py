from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torchvision.transforms import Compose, Resize 
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo

from . import models_vitmm

from .util.patch_embed import PatchEmbed_new
from timm.models.layers import to_2tuple

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                checkpoint path for loading pretrained weights.

            model_config:
                config path for your model.
        """
        super().__init__()

        av_fusion = True
        ckpt = torch.load('/work/u8090533/mavil/mavil_as_pt_ft_a+v.pth', map_location='cpu')
        self.model = models_vitmm.vitmm_base_patch16(
                    num_classes=527,
                    drop_path_rate=0.1,
                    global_pool=True,
                    mask_2d=True,
                    av_fusion=av_fusion, 
                    depth_av=3 if av_fusion else 0,
                    n_frm=8, # 8 frames per video
                    pos_train=False,
                )

        img_size = (1024, 128) # 1024, 128
        in_chans = 1
        emb_dim = 768
        self.model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=in_chans, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        num_patches = self.model.patch_embed.num_patches
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        
        checkpoint_model = ckpt['model']
        self.model.load_state_dict(checkpoint_model, strict=True)

        self.audio_conf = {'sample_rate': 16000,
                    'num_mel_bins': 128,
                    'target_length': 1024,  # crop/pad spectogram to roughly 10 seconds
                    # 'mean': -4.2677393,     # precomputed on audioset
                    # 'std': 4.5689974,       # precomputed on audioset
                    'noise': False,
                    }
        
        self.video_frame_size = (224, 224)
        self.video_len = 4                  # MAViL takes 4 sec clips
        self.video_frame_rate = 2           # at 2 frames per second

        self.video_transform = Compose([
            ToTensorVideo(),
            Resize((224,224)),
            NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def preprocess_video(self, video, video_frame_rate):
        # 1. Resample video
        # (from https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/datasets/video_utils.py#L278)
        step = float(video_frame_rate) / self.video_frame_rate
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            idxs = slice(None, None, step)
        else:
            num_frames = int(len(video) / step)
            idxs = torch.arange(num_frames, dtype=torch.float32) * step
            idxs = idxs.floor().to(torch.int64)
        video = video[idxs]
        
        # 2. TCHW -> THWC and crop/repeat to four sec.
        video = torch.permute(video, (0,2,3,1))
        n_target_frm = self.video_len * self.video_frame_rate       # 8 frames per clip
        if video.shape[0] < n_target_frm:
            n_repeats = n_target_frm//video.shape[0] + 1
            video = video.repeat(n_repeats,1,1,1)
        video = video[:8]

        preprocessed_video = self.video_transform(video)
        return preprocessed_video
    
    def preprocess_audio(self, audio, audio_sample_rate, fbank_mean=None, fbank_std=None):
        if len(audio.shape) == 2:
            audio = audio.mean(dim=0)
        waveform = audio.unsqueeze(0)
        waveform = waveform - waveform.mean()

        # Audio resampling done by fbank

        # 498x128, 998x128, 1000x128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=audio_sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.audio_conf.get('num_mel_bins'), dither=0.0, frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        # fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        # fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq

        # normalize fbank (allow precomputed mean/std)
        # on-the-fly normalization causes 6% mAP drop on subset of Audioset
        if fbank_mean is not None and fbank_std is not None:
            fbank = (fbank - fbank_mean) / (fbank_std * 2) 
        else:
            fbank = (fbank - fbank.mean()) / (fbank.std() * 2) 

        return fbank.unsqueeze(0)

    def forward(
        self, source: List[Tuple[Tensor, Tensor]]
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Replace this function run a forward pass with your model
        source: list of audio-video Tensor tuples
                [(wav1,vid1), (wav2,vid2), ...]
                in your input format
        """
        audio, video = zip(*source)
        x = torch.stack(audio, dim=0)
        v = torch.stack(video, dim=0)
        v = v[:,:,:self.model.n_frm,:,:]

        B = x.shape[0]

        # audio
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, 1:, :]
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.pos_drop(x)

        # audio encoding
        audio_seq_feats = []
        audio_pooled_feats = []
        for blk in self.model.blocks:
            audio_seq_feats.append(x[:, 1:, :]) # drop CLS token
            audio_pooled_feats.append(x[:, 1:, :].mean(dim=1, keepdim=True)) # drop CLS token and pool
            x = blk(x)

        audio_seq_feats.append(x[:, 1:, :]) # drop CLS token
        audio_pooled_feats.append(x[:, 1:, :].mean(dim=1, keepdim=True)) # drop CLS token and pool

        # video
        v = self.model.patch_embed_v(v)
        v = v + self.model.pos_embed_v[:, 1:, :]
        cls_token_v = self.model.cls_token_v + self.model.pos_embed_v[:, :1, :]
        cls_tokens_v = cls_token_v.expand(B, -1, -1)
        v = torch.cat((cls_tokens_v, v), dim=1)
        v = self.model.pos_drop(v)

        # video encoding
        video_seq_feats = []
        video_pooled_feats = []
        for blk in self.model.blocks_v:
            video_seq_feats.append(v[:, 1:, :]) # drop CLS token
            video_pooled_feats.append(v[:, 1:, :].mean(dim=1, keepdim=True)) # drop CLS token and pool
            v = blk(v)
        
        video_seq_feats.append(v[:, 1:, :]) # drop CLS token
        video_pooled_feats.append(v[:, 1:, :].mean(dim=1, keepdim=True)) # drop CLS token and pool


        if self.model.av_fusion:
            x_len = x.shape[1]
            xv = torch.cat((x,v), dim=1)

            fusion_seq_feats = []
            fusion_pooled_feats = []
            for blk in self.model.blocks_av:
                
                fusion_seq_feats.append(xv) 
                x = xv[:, 1:x_len, :].mean(dim=1, keepdim=True)  # global pool without cls token
                v = xv[:, x_len+1:,:].mean(dim=1, keepdim=True) # global pool without cls token
                fusion_pooled_feats.append(torch.cat((x,v),dim=2))

                xv =  blk(xv)

            fusion_seq_feats.append(xv) 
            x = xv[:, 1:x_len, :].mean(dim=1, keepdim=True)  # global pool without cls token
            v = xv[:, x_len+1:,:].mean(dim=1, keepdim=True) # global pool without cls token
            fusion_pooled_feats.append(self.model.fc_norm_av(torch.cat((x,v),dim=2)))

        # Return intermediate layer representations for potential layer-wise experiments
        # Dict should contain three items, with keys as listed below:
        # video_feats: features that only use visual modality as input
        # audio_feats: features that only use auditory modality as input
        # fusion_feats: features that consider both modalities
        # Each item should be a list of features that are of the same shape
        return {
            "video_feats": video_pooled_feats,
            "audio_feats": audio_pooled_feats,
            "fusion_feats": fusion_pooled_feats,
            "video_seq_feats": video_seq_feats,
            "audio_seq_feats": audio_seq_feats,
            "fusion_seq_feats": fusion_seq_feats,
        }
