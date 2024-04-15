import yaml
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as aT
import torchvision
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from interfaces import UpstreamBase

from skimage.feature import hog

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        self.video_frame_size = (128, 64)
        self.video_frame_rate = 25

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def preprocess_video(self, video, video_frame_rate):
        
        # Resize video frames
        video_frames = []
        for frame in video:
            video_frames.append(
                torchvision.transforms.functional.resize(
                    frame, self.video_frame_size, antialias=False
                )
            )
        video = torch.stack(video_frames)

        # Resample video
        # (from https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/datasets/video_utils.py#L278)
        step = float(video_frame_rate) / self.video_frame_rate
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            idxs = slice(None, None, step)
        else:
            num_frames = max(int(len(video) / step), 1)
            idxs = torch.arange(num_frames, dtype=torch.float32) * step
            idxs = idxs.floor().to(torch.int64)
        video = video[idxs]

        if video.shape[1] == 3:
            video = 0.2989 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]
        else:
            video = video.mean(dim=1)

        return video
    
    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length) or (audio_length,), where audio_channels is usually 1 or 2
        """
        if len(audio.shape) == 2:
            audio = audio[0]

        return audio[0]
    
    def forward(self, source):

        audio, video = zip(*source)

        # TODO: add here, try out hyperparams that work decently
        # 
        # ( Add moving object detection if useful, as shown in Section 3.1: 
        # https://www.mdpi.com/1424-8220/20/24/7299 )
        #

        videos = pad_sequence(video, batch_first = True)

        video_feats = []
        for video in videos:
            video_feat = []
            for frame in video:
                feat = hog(frame.cpu(), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                feat = torch.from_numpy(feat)
                video_feat.append(feat)
            video_feat = torch.stack(video_feat)
            video_feats.append(video_feat)
        video_feats = torch.stack(video_feats).cuda()

        return {
            "video_feats": [video_feats],
            "audio_feats": [],
            "fusion_feats": [],
        }
