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

        self.video_frame_rate = 25

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def preprocess_video(self, video, video_frame_rate):

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

        return video
    
    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
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
        videos = videos.permute(0, 1, 3, 4, 2)

        print(videos.shape)

        video_feats = []
        for video in videos:
            video_feat = []
            for frame in video:
                feat = hog(frame.cpu(), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), channel_axis=-1)
                feat = torch.from_numpy(feat)
                video_feat.append(feat)
            video_feat = torch.stack(video_feat)
            video_feats.append(video_feat)
        video_feats = torch.stack(video_feats)

        print(video_feats.shape)

        return {
            "video_feats": [video_feats],
            "audio_feats": [],
            "fusion_feats": [],
        }
