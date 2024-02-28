import torch
import hub
from torchvision.io import read_video
from utils.preprocess import preprocess

device = "cuda:0"

fname = "/media/rogert/DATA/AudioSet-20K/eval/video_mp4_288p/007P6bFgRCU.mp4"

# Load pretrained audio-visual model
# check hubconf.py files in upstream_models directory for all available models
model = getattr(hub, 'avbert')()
model.eval()
model.to(device)

# frames:   video frames,   temporal length x RGB channels x height x width
# wav:      audio waveform, audio channels x temporal length
# meta:     dict with video frame rate and audio sample rate
frames, wav, meta = read_video(fname, pts_unit="sec", output_format="TCHW")
video_fps = meta["video_fps"]
audio_sr = meta["audio_fps"]

# print(frames.shape) 
# print(frames.dtype) # torch.uint8
# print(wav.shape)
# print(wav.dtype) # torch.float32
# print(meta)

# Model expects preprocessed (audio, video) pairs
# pairs: list of (prepped_wav, prepped_frames) tuples
pairs = [
    preprocess(model, frames, wav, video_fps, audio_sr, device)
]

# Feature extraction
# features: dict of audio, video, and fusion features,
#           each feature is a list of Tensors, 
#           one per Transformer layer in the respective encoder
features = model(pairs)
print(features)