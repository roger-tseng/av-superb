import models_vitmm
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

print("Finished Imports...")
av_fusion = False
ckpt = torch.load("/home/u7196393/mavil/mavil_as_pt_ft_a_v.pth", map_location="cpu")
model = models_vitmm.vitmm_base_patch16(
    num_classes=527,
    drop_path_rate=0.1,
    global_pool=True,
    mask_2d=True,
    av_fusion=av_fusion,
    depth_av=3 if av_fusion else 0,
    n_frm=8,  # 8 frames per video
    pos_train=False,
)
print("Finished Model Definition...")


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches

        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


print("Finished New Patch...")
img_size = (1024, 128)  # 1024, 128
in_chans = 1
emb_dim = 768
model.patch_embed = PatchEmbed_new(
    img_size=img_size,
    patch_size=(16, 16),
    in_chans=in_chans,
    embed_dim=emb_dim,
    stride=16,
)  # no overlap. stride=img_size=16
num_patches = model.patch_embed.num_patches
model.pos_embed = nn.Parameter(
    torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False
)  # fixed sin-cos embedding

checkpoint_model = ckpt["model"]
state_dict = model.state_dict()

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=True)
print(msg)

model.cuda()
print()
print("end here!")
