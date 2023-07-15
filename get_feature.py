import torch
import hub
import torchvision

device = "cuda:0"

model = getattr(hub, 'avhubert')()
model.eval()

frames, wav, meta = torchvision.io.read_video('/livingrooms/rogertseng/lrs3/trainval/ZzugJPASNB8/50011.mp4', pts_unit="sec", output_format="TCHW")
print(frames.shape)
print(wav.shape)

prep_a = model.preprocess_audio(wav, meta["audio_fps"]).to(device)
prep_v = model.preprocess_video(frames, meta["video_fps"]).to(device)
print(prep_a.shape)
print(prep_v.shape)

pair = [(prep_a, prep_v) for i in range(64)]
import time
for i in range(500):
    time.sleep(1)
    model = model.to(device)
    features = model(pair)
l0 = features['fusion_feats'][0][0]

actual = torch.load('/home/rogertseng/audiovisual-benchmark/features/avhubert_fusion_feats/_livingrooms_rogertseng_lrs3_trainval_ZzugJPASNB8_50011_pooled.pt')
# actual.shape
# len(actual)
actual_l0 = actual[0]
print(actual_l0[-1])
print(l0[-1])
