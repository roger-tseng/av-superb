import hub 
import torch

model = getattr(hub, 'avbert')()

# for i in range(5,10):
#     length = i/5
#     audio_samples = length * 16000
#     video_samples = length * 25
#     m = torch.nn.ConstantPad1d((0,16385-int(audio_samples)), 0)
#     wav = m(torch.randn(int(audio_samples)))

#     video = torch.ones(int(video_samples), 3, 112, 112, dtype=torch.uint8)
#     print("i = ", i)
#     out = model.preprocess(video, wav, 25, 16000)

length = 5
audio_samples = length * 16000
video_samples = length * 25
m = torch.nn.ConstantPad1d((0,16385-int(audio_samples)), 0)
wav = m(torch.randn(int(audio_samples)))
video = torch.ones(int(video_samples), 3, 112, 112, dtype=torch.uint8)
out = model.preprocess(video, wav, 25, 16000)
features = model([(out[1], out[0])])
print("hi")