# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/baseline/extracter.py ]
#   Synopsis     [ the baseline wrapper ]
#   Author       [ S3PRL (Leo Yang) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import yaml
from torch.nn.utils.rnn import pad_sequence

from interfaces import UpstreamBase
from .extracter import get_extracter
import torchaudio
# from .preprocessor import get_preprocessor

SAMPLE_RATE = 16000

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi or torchaudio preprocessor
    Support: fbank
    """

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        self.audio_sample_rate = 16000
        
        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        if "kaldi" in self.config:
            self.extracter, self.output_dim, frame_shift = get_extracter(self.config)
            self.downsample_rate = round(frame_shift * SAMPLE_RATE / 1000)
        else:
            raise NotImplementedError
            # self.extracter, self.output_dim, _ = get_preprocessor(
            #     self.config, process_input_only=True
            # )
            # self.downsample_rate = round(
            #     self.config.get("hop_ms", 10) * SAMPLE_RATE / 1000
            # )

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))
        return feats

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def preprocess_video(self, video, video_frame_rate):
        return video[0][0][0]
    
    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        """
        if len(audio.shape) == 2:
            audio = audio.mean(dim=0)

        # Resample audio
        if audio_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sample_rate, self.audio_sample_rate
            )

        return audio
    
    def forward(self, source):

        wavs, video = zip(*source)

        if "kaldi" in self.config:
            feats = self._extractor_forward(wavs)
        else:
            raise NotImplementedError

        padded_feats = pad_sequence(feats, batch_first=True)
        return {
            "audio_feats": [padded_feats],
        }