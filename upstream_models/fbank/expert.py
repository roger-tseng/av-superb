from torch.nn.utils.rnn import pad_sequence
from interfaces import UpstreamBase
from .extracter import get_extracter

# kaldi:
#   feat_type: fbank
#   fbank:
#     num_mel_bins: 80
#     frame_length: 25.0
#     frame_shift: 10.0
#     use_log_fbank: True

# delta:
#   order: 2
#   win_length: 5

# cmvn:
#   use_cmvn: False

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
    
        self.config = {'kaldi': {'feat_type': 'fbank', 'fbank': {'num_mel_bins': 80, 'frame_length': 25.0, 'frame_shift': 10.0, 'use_log_fbank': True}}, 'delta': {'order': 2, 'win_length': 5}, 'cmvn': {'use_cmvn': False}}
        self.extracter, self.output_dim, frame_shift = get_extracter(self.config)

    def preprocess_video(self, video, video_frame_rate):
        return video[0][0][0]
    
    def preprocess_audio(self, audio, audio_sample_rate):
        return self.extracter(wav)
    
    def forward(self, source):

        wavs, video = zip(*source)

        return wav


