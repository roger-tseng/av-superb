import yaml
from torch.nn.utils.rnn import pad_sequence

from interfaces import UpstreamBase

from skimage.feature import hog

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    # def get_downsample_rates(self, key: str) -> int:
    #     return self.downsample_rate

    def preprocess_video(self, video, video_frame_rate):
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

        wavs, video = zip(*source)

        # TODO: add here, try out hyperparams that work decently
        # 
        # ( Add moving object detection if useful, as shown in Section 3.1: 
        # https://www.mdpi.com/1424-8220/20/24/7299 )
        # 
        # feats = hog(image, orientations=8, pixels_per_cell=(16, 16),
        #                     cells_per_block=(1, 1), channel_axis=-1)

        return {
            "video_feats": [None],
        }