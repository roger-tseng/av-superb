# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


vision_list=['patch_embed_v.proj.weight','patch_embed_v.proj.bias',
'blocks_v.0.norm1.weight','blocks_v.0.norm1.bias','blocks_v.0.attn.qkv.weight','blocks_v.0.attn.qkv.bias','blocks_v.0.attn.proj.weight','blocks_v.0.attn.proj.bias','blocks_v.0.norm2.weight','blocks_v.0.norm2.bias','blocks_v.0.mlp.fc1.weight','blocks_v.0.mlp.fc1.bias','blocks_v.0.mlp.fc2.weight','blocks_v.0.mlp.fc2.bias',
'blocks_v.1.norm1.weight','blocks_v.1.norm1.bias','blocks_v.1.attn.qkv.weight','blocks_v.1.attn.qkv.bias','blocks_v.1.attn.proj.weight','blocks_v.1.attn.proj.bias','blocks_v.1.norm2.weight','blocks_v.1.norm2.bias','blocks_v.1.mlp.fc1.weight','blocks_v.1.mlp.fc1.bias','blocks_v.1.mlp.fc2.weight','blocks_v.1.mlp.fc2.bias',
'blocks_v.2.norm1.weight','blocks_v.2.norm1.bias','blocks_v.2.attn.qkv.weight','blocks_v.2.attn.qkv.bias','blocks_v.2.attn.proj.weight','blocks_v.2.attn.proj.bias','blocks_v.2.norm2.weight','blocks_v.2.norm2.bias','blocks_v.2.mlp.fc1.weight','blocks_v.2.mlp.fc1.bias','blocks_v.2.mlp.fc2.weight','blocks_v.2.mlp.fc2.bias',
'blocks_v.3.norm1.weight','blocks_v.3.norm1.bias','blocks_v.3.attn.qkv.weight','blocks_v.3.attn.qkv.bias','blocks_v.3.attn.proj.weight','blocks_v.3.attn.proj.bias','blocks_v.3.norm2.weight','blocks_v.3.norm2.bias','blocks_v.3.mlp.fc1.weight','blocks_v.3.mlp.fc1.bias','blocks_v.3.mlp.fc2.weight','blocks_v.3.mlp.fc2.bias',
'blocks_v.4.norm1.weight','blocks_v.4.norm1.bias','blocks_v.4.attn.qkv.weight','blocks_v.4.attn.qkv.bias','blocks_v.4.attn.proj.weight','blocks_v.4.attn.proj.bias','blocks_v.4.norm2.weight','blocks_v.4.norm2.bias','blocks_v.4.mlp.fc1.weight','blocks_v.4.mlp.fc1.bias','blocks_v.4.mlp.fc2.weight','blocks_v.4.mlp.fc2.bias',
'blocks_v.5.norm1.weight','blocks_v.5.norm1.bias','blocks_v.5.attn.qkv.weight','blocks_v.5.attn.qkv.bias','blocks_v.5.attn.proj.weight','blocks_v.5.attn.proj.bias','blocks_v.5.norm2.weight','blocks_v.5.norm2.bias','blocks_v.5.mlp.fc1.weight','blocks_v.5.mlp.fc1.bias','blocks_v.5.mlp.fc2.weight','blocks_v.5.mlp.fc2.bias',
'blocks_v.6.norm1.weight','blocks_v.6.norm1.bias','blocks_v.6.attn.qkv.weight','blocks_v.6.attn.qkv.bias','blocks_v.6.attn.proj.weight','blocks_v.6.attn.proj.bias','blocks_v.6.norm2.weight','blocks_v.6.norm2.bias','blocks_v.6.mlp.fc1.weight','blocks_v.6.mlp.fc1.bias','blocks_v.6.mlp.fc2.weight','blocks_v.6.mlp.fc2.bias',
'blocks_v.7.norm1.weight','blocks_v.7.norm1.bias','blocks_v.7.attn.qkv.weight','blocks_v.7.attn.qkv.bias','blocks_v.7.attn.proj.weight','blocks_v.7.attn.proj.bias','blocks_v.7.norm2.weight','blocks_v.7.norm2.bias','blocks_v.7.mlp.fc1.weight','blocks_v.7.mlp.fc1.bias','blocks_v.7.mlp.fc2.weight','blocks_v.7.mlp.fc2.bias',
'blocks_v.8.norm1.weight','blocks_v.8.norm1.bias','blocks_v.8.attn.qkv.weight','blocks_v.8.attn.qkv.bias','blocks_v.8.attn.proj.weight','blocks_v.8.attn.proj.bias','blocks_v.8.norm2.weight','blocks_v.8.norm2.bias','blocks_v.8.mlp.fc1.weight','blocks_v.8.mlp.fc1.bias','blocks_v.8.mlp.fc2.weight','blocks_v.8.mlp.fc2.bias',
'blocks_v.9.norm1.weight','blocks_v.9.norm1.bias','blocks_v.9.attn.qkv.weight','blocks_v.9.attn.qkv.bias','blocks_v.9.attn.proj.weight','blocks_v.9.attn.proj.bias','blocks_v.9.norm2.weight','blocks_v.9.norm2.bias','blocks_v.9.mlp.fc1.weight','blocks_v.9.mlp.fc1.bias','blocks_v.9.mlp.fc2.weight','blocks_v.9.mlp.fc2.bias',
'blocks_v.10.norm1.weight','blocks_v.10.norm1.bias','blocks_v.10.attn.qkv.weight','blocks_v.10.attn.qkv.bias','blocks_v.10.attn.proj.weight','blocks_v.10.attn.proj.bias','blocks_v.10.norm2.weight','blocks_v.10.norm2.bias','blocks_v.10.mlp.fc1.weight','blocks_v.10.mlp.fc1.bias','blocks_v.10.mlp.fc2.weight','blocks_v.10.mlp.fc2.bias',
'blocks_v.11.norm1.weight','blocks_v.11.norm1.bias','blocks_v.11.attn.qkv.weight','blocks_v.11.attn.qkv.bias','blocks_v.11.attn.proj.weight','blocks_v.11.attn.proj.bias','blocks_v.11.norm2.weight','blocks_v.11.norm2.bias','blocks_v.11.mlp.fc1.weight','blocks_v.11.mlp.fc1.bias','blocks_v.11.mlp.fc2.weight','blocks_v.11.mlp.fc2.bias',
'blocks_v.12.norm1.weight','blocks_v.12.norm1.bias','blocks_v.12.attn.qkv.weight','blocks_v.12.attn.qkv.bias','blocks_v.12.attn.proj.weight','blocks_v.12.attn.proj.bias','blocks_v.12.norm2.weight','blocks_v.12.norm2.bias','blocks_v.12.mlp.fc1.weight','blocks_v.12.mlp.fc1.bias','blocks_v.12.mlp.fc2.weight','blocks_v.12.mlp.fc2.bias',
'blocks_v.13.norm1.weight','blocks_v.13.norm1.bias','blocks_v.13.attn.qkv.weight','blocks_v.13.attn.qkv.bias','blocks_v.13.attn.proj.weight','blocks_v.13.attn.proj.bias','blocks_v.13.norm2.weight','blocks_v.13.norm2.bias','blocks_v.13.mlp.fc1.weight','blocks_v.13.mlp.fc1.bias','blocks_v.13.mlp.fc2.weight','blocks_v.13.mlp.fc2.bias',
'blocks_v.14.norm1.weight','blocks_v.14.norm1.bias','blocks_v.14.attn.qkv.weight','blocks_v.14.attn.qkv.bias','blocks_v.14.attn.proj.weight','blocks_v.14.attn.proj.bias','blocks_v.14.norm2.weight','blocks_v.14.norm2.bias','blocks_v.14.mlp.fc1.weight','blocks_v.14.mlp.fc1.bias','blocks_v.14.mlp.fc2.weight','blocks_v.14.mlp.fc2.bias',
'blocks_v.15.norm1.weight','blocks_v.15.norm1.bias','blocks_v.15.attn.qkv.weight','blocks_v.15.attn.qkv.bias','blocks_v.15.attn.proj.weight','blocks_v.15.attn.proj.bias','blocks_v.15.norm2.weight','blocks_v.15.norm2.bias','blocks_v.15.mlp.fc1.weight','blocks_v.15.mlp.fc1.bias','blocks_v.15.mlp.fc2.weight','blocks_v.15.mlp.fc2.bias',
'blocks_v.16.norm1.weight','blocks_v.16.norm1.bias','blocks_v.16.attn.qkv.weight','blocks_v.16.attn.qkv.bias','blocks_v.16.attn.proj.weight','blocks_v.16.attn.proj.bias','blocks_v.16.norm2.weight','blocks_v.16.norm2.bias','blocks_v.16.mlp.fc1.weight','blocks_v.16.mlp.fc1.bias','blocks_v.16.mlp.fc2.weight','blocks_v.16.mlp.fc2.bias',
'blocks_v.17.norm1.weight','blocks_v.17.norm1.bias','blocks_v.17.attn.qkv.weight','blocks_v.17.attn.qkv.bias','blocks_v.17.attn.proj.weight','blocks_v.17.attn.proj.bias','blocks_v.17.norm2.weight','blocks_v.17.norm2.bias','blocks_v.17.mlp.fc1.weight','blocks_v.17.mlp.fc1.bias','blocks_v.17.mlp.fc2.weight','blocks_v.17.mlp.fc2.bias',
'blocks_v.18.norm1.weight','blocks_v.18.norm1.bias','blocks_v.18.attn.qkv.weight','blocks_v.18.attn.qkv.bias','blocks_v.18.attn.proj.weight','blocks_v.18.attn.proj.bias','blocks_v.18.norm2.weight','blocks_v.18.norm2.bias','blocks_v.18.mlp.fc1.weight','blocks_v.18.mlp.fc1.bias','blocks_v.18.mlp.fc2.weight','blocks_v.18.mlp.fc2.bias',
'blocks_v.19.norm1.weight','blocks_v.19.norm1.bias','blocks_v.19.attn.qkv.weight','blocks_v.19.attn.qkv.bias','blocks_v.19.attn.proj.weight','blocks_v.19.attn.proj.bias','blocks_v.19.norm2.weight','blocks_v.19.norm2.bias','blocks_v.19.mlp.fc1.weight','blocks_v.19.mlp.fc1.bias','blocks_v.19.mlp.fc2.weight','blocks_v.19.mlp.fc2.bias',
'blocks_v.20.norm1.weight','blocks_v.20.norm1.bias','blocks_v.20.attn.qkv.weight','blocks_v.20.attn.qkv.bias','blocks_v.20.attn.proj.weight','blocks_v.20.attn.proj.bias','blocks_v.20.norm2.weight','blocks_v.20.norm2.bias','blocks_v.20.mlp.fc1.weight','blocks_v.20.mlp.fc1.bias','blocks_v.20.mlp.fc2.weight','blocks_v.20.mlp.fc2.bias',
'blocks_v.21.norm1.weight','blocks_v.21.norm1.bias','blocks_v.21.attn.qkv.weight','blocks_v.21.attn.qkv.bias','blocks_v.21.attn.proj.weight','blocks_v.21.attn.proj.bias','blocks_v.21.norm2.weight','blocks_v.21.norm2.bias','blocks_v.21.mlp.fc1.weight','blocks_v.21.mlp.fc1.bias','blocks_v.21.mlp.fc2.weight','blocks_v.21.mlp.fc2.bias',
'blocks_v.22.norm1.weight','blocks_v.22.norm1.bias','blocks_v.22.attn.qkv.weight','blocks_v.22.attn.qkv.bias','blocks_v.22.attn.proj.weight','blocks_v.22.attn.proj.bias','blocks_v.22.norm2.weight','blocks_v.22.norm2.bias','blocks_v.22.mlp.fc1.weight','blocks_v.22.mlp.fc1.bias','blocks_v.22.mlp.fc2.weight','blocks_v.22.mlp.fc2.bias',
'blocks_v.23.norm1.weight','blocks_v.23.norm1.bias','blocks_v.23.attn.qkv.weight','blocks_v.23.attn.qkv.bias','blocks_v.23.attn.proj.weight','blocks_v.23.attn.proj.bias','blocks_v.23.norm2.weight','blocks_v.23.norm2.bias','blocks_v.23.mlp.fc1.weight','blocks_v.23.mlp.fc1.bias','blocks_v.23.mlp.fc2.weight','blocks_v.23.mlp.fc2.bias',
]
audio_video_list=[
'patch_embed_v.proj.weight','patch_embed_v.proj.bias', 'pos_emb_v'
'blocks_v.0.norm1.weight','blocks_v.0.norm1.bias','blocks_v.0.attn.qkv.weight','blocks_v.0.attn.qkv.bias','blocks_v.0.attn.proj.weight','blocks_v.0.attn.proj.bias','blocks_v.0.norm2.weight','blocks_v.0.norm2.bias','blocks_v.0.mlp.fc1.weight','blocks_v.0.mlp.fc1.bias','blocks_v.0.mlp.fc2.weight','blocks_v.0.mlp.fc2.bias',
'blocks_v.1.norm1.weight','blocks_v.1.norm1.bias','blocks_v.1.attn.qkv.weight','blocks_v.1.attn.qkv.bias','blocks_v.1.attn.proj.weight','blocks_v.1.attn.proj.bias','blocks_v.1.norm2.weight','blocks_v.1.norm2.bias','blocks_v.1.mlp.fc1.weight','blocks_v.1.mlp.fc1.bias','blocks_v.1.mlp.fc2.weight','blocks_v.1.mlp.fc2.bias',
'blocks_v.2.norm1.weight','blocks_v.2.norm1.bias','blocks_v.2.attn.qkv.weight','blocks_v.2.attn.qkv.bias','blocks_v.2.attn.proj.weight','blocks_v.2.attn.proj.bias','blocks_v.2.norm2.weight','blocks_v.2.norm2.bias','blocks_v.2.mlp.fc1.weight','blocks_v.2.mlp.fc1.bias','blocks_v.2.mlp.fc2.weight','blocks_v.2.mlp.fc2.bias',
'blocks_v.3.norm1.weight','blocks_v.3.norm1.bias','blocks_v.3.attn.qkv.weight','blocks_v.3.attn.qkv.bias','blocks_v.3.attn.proj.weight','blocks_v.3.attn.proj.bias','blocks_v.3.norm2.weight','blocks_v.3.norm2.bias','blocks_v.3.mlp.fc1.weight','blocks_v.3.mlp.fc1.bias','blocks_v.3.mlp.fc2.weight','blocks_v.3.mlp.fc2.bias',
'blocks_v.4.norm1.weight','blocks_v.4.norm1.bias','blocks_v.4.attn.qkv.weight','blocks_v.4.attn.qkv.bias','blocks_v.4.attn.proj.weight','blocks_v.4.attn.proj.bias','blocks_v.4.norm2.weight','blocks_v.4.norm2.bias','blocks_v.4.mlp.fc1.weight','blocks_v.4.mlp.fc1.bias','blocks_v.4.mlp.fc2.weight','blocks_v.4.mlp.fc2.bias',
'blocks_v.5.norm1.weight','blocks_v.5.norm1.bias','blocks_v.5.attn.qkv.weight','blocks_v.5.attn.qkv.bias','blocks_v.5.attn.proj.weight','blocks_v.5.attn.proj.bias','blocks_v.5.norm2.weight','blocks_v.5.norm2.bias','blocks_v.5.mlp.fc1.weight','blocks_v.5.mlp.fc1.bias','blocks_v.5.mlp.fc2.weight','blocks_v.5.mlp.fc2.bias',
'blocks_v.6.norm1.weight','blocks_v.6.norm1.bias','blocks_v.6.attn.qkv.weight','blocks_v.6.attn.qkv.bias','blocks_v.6.attn.proj.weight','blocks_v.6.attn.proj.bias','blocks_v.6.norm2.weight','blocks_v.6.norm2.bias','blocks_v.6.mlp.fc1.weight','blocks_v.6.mlp.fc1.bias','blocks_v.6.mlp.fc2.weight','blocks_v.6.mlp.fc2.bias',
'blocks_v.7.norm1.weight','blocks_v.7.norm1.bias','blocks_v.7.attn.qkv.weight','blocks_v.7.attn.qkv.bias','blocks_v.7.attn.proj.weight','blocks_v.7.attn.proj.bias','blocks_v.7.norm2.weight','blocks_v.7.norm2.bias','blocks_v.7.mlp.fc1.weight','blocks_v.7.mlp.fc1.bias','blocks_v.7.mlp.fc2.weight','blocks_v.7.mlp.fc2.bias',
'blocks_v.8.norm1.weight','blocks_v.8.norm1.bias','blocks_v.8.attn.qkv.weight','blocks_v.8.attn.qkv.bias','blocks_v.8.attn.proj.weight','blocks_v.8.attn.proj.bias','blocks_v.8.norm2.weight','blocks_v.8.norm2.bias','blocks_v.8.mlp.fc1.weight','blocks_v.8.mlp.fc1.bias','blocks_v.8.mlp.fc2.weight','blocks_v.8.mlp.fc2.bias',
'blocks_v.9.norm1.weight','blocks_v.9.norm1.bias','blocks_v.9.attn.qkv.weight','blocks_v.9.attn.qkv.bias','blocks_v.9.attn.proj.weight','blocks_v.9.attn.proj.bias','blocks_v.9.norm2.weight','blocks_v.9.norm2.bias','blocks_v.9.mlp.fc1.weight','blocks_v.9.mlp.fc1.bias','blocks_v.9.mlp.fc2.weight','blocks_v.9.mlp.fc2.bias',
'blocks_v.10.norm1.weight','blocks_v.10.norm1.bias','blocks_v.10.attn.qkv.weight','blocks_v.10.attn.qkv.bias','blocks_v.10.attn.proj.weight','blocks_v.10.attn.proj.bias','blocks_v.10.norm2.weight','blocks_v.10.norm2.bias','blocks_v.10.mlp.fc1.weight','blocks_v.10.mlp.fc1.bias','blocks_v.10.mlp.fc2.weight','blocks_v.10.mlp.fc2.bias',
'blocks_v.11.norm1.weight','blocks_v.11.norm1.bias','blocks_v.11.attn.qkv.weight','blocks_v.11.attn.qkv.bias','blocks_v.11.attn.proj.weight','blocks_v.11.attn.proj.bias','blocks_v.11.norm2.weight','blocks_v.11.norm2.bias','blocks_v.11.mlp.fc1.weight','blocks_v.11.mlp.fc1.bias','blocks_v.11.mlp.fc2.weight','blocks_v.11.mlp.fc2.bias'
'blocks_v.12.norm1.weight','blocks_v.12.norm1.bias','blocks_v.12.attn.qkv.weight','blocks_v.12.attn.qkv.bias','blocks_v.12.attn.proj.weight','blocks_v.12.attn.proj.bias','blocks_v.12.norm2.weight','blocks_v.12.norm2.bias','blocks_v.12.mlp.fc1.weight','blocks_v.12.mlp.fc1.bias','blocks_v.12.mlp.fc2.weight','blocks_v.12.mlp.fc2.bias',
'blocks_v.13.norm1.weight','blocks_v.13.norm1.bias','blocks_v.13.attn.qkv.weight','blocks_v.13.attn.qkv.bias','blocks_v.13.attn.proj.weight','blocks_v.13.attn.proj.bias','blocks_v.13.norm2.weight','blocks_v.13.norm2.bias','blocks_v.13.mlp.fc1.weight','blocks_v.13.mlp.fc1.bias','blocks_v.13.mlp.fc2.weight','blocks_v.13.mlp.fc2.bias',
'blocks_v.14.norm1.weight','blocks_v.14.norm1.bias','blocks_v.14.attn.qkv.weight','blocks_v.14.attn.qkv.bias','blocks_v.14.attn.proj.weight','blocks_v.14.attn.proj.bias','blocks_v.14.norm2.weight','blocks_v.14.norm2.bias','blocks_v.14.mlp.fc1.weight','blocks_v.14.mlp.fc1.bias','blocks_v.14.mlp.fc2.weight','blocks_v.14.mlp.fc2.bias',
'blocks_v.15.norm1.weight','blocks_v.15.norm1.bias','blocks_v.15.attn.qkv.weight','blocks_v.15.attn.qkv.bias','blocks_v.15.attn.proj.weight','blocks_v.15.attn.proj.bias','blocks_v.15.norm2.weight','blocks_v.15.norm2.bias','blocks_v.15.mlp.fc1.weight','blocks_v.15.mlp.fc1.bias','blocks_v.15.mlp.fc2.weight','blocks_v.15.mlp.fc2.bias',
'blocks_v.16.norm1.weight','blocks_v.16.norm1.bias','blocks_v.16.attn.qkv.weight','blocks_v.16.attn.qkv.bias','blocks_v.16.attn.proj.weight','blocks_v.16.attn.proj.bias','blocks_v.16.norm2.weight','blocks_v.16.norm2.bias','blocks_v.16.mlp.fc1.weight','blocks_v.16.mlp.fc1.bias','blocks_v.16.mlp.fc2.weight','blocks_v.16.mlp.fc2.bias',
'blocks_v.17.norm1.weight','blocks_v.17.norm1.bias','blocks_v.17.attn.qkv.weight','blocks_v.17.attn.qkv.bias','blocks_v.17.attn.proj.weight','blocks_v.17.attn.proj.bias','blocks_v.17.norm2.weight','blocks_v.17.norm2.bias','blocks_v.17.mlp.fc1.weight','blocks_v.17.mlp.fc1.bias','blocks_v.17.mlp.fc2.weight','blocks_v.17.mlp.fc2.bias',
'blocks_v.18.norm1.weight','blocks_v.18.norm1.bias','blocks_v.18.attn.qkv.weight','blocks_v.18.attn.qkv.bias','blocks_v.18.attn.proj.weight','blocks_v.18.attn.proj.bias','blocks_v.18.norm2.weight','blocks_v.18.norm2.bias','blocks_v.18.mlp.fc1.weight','blocks_v.18.mlp.fc1.bias','blocks_v.18.mlp.fc2.weight','blocks_v.18.mlp.fc2.bias',
'blocks_v.19.norm1.weight','blocks_v.19.norm1.bias','blocks_v.19.attn.qkv.weight','blocks_v.19.attn.qkv.bias','blocks_v.19.attn.proj.weight','blocks_v.19.attn.proj.bias','blocks_v.19.norm2.weight','blocks_v.19.norm2.bias','blocks_v.19.mlp.fc1.weight','blocks_v.19.mlp.fc1.bias','blocks_v.19.mlp.fc2.weight','blocks_v.19.mlp.fc2.bias',
'blocks_v.20.norm1.weight','blocks_v.20.norm1.bias','blocks_v.20.attn.qkv.weight','blocks_v.20.attn.qkv.bias','blocks_v.20.attn.proj.weight','blocks_v.20.attn.proj.bias','blocks_v.20.norm2.weight','blocks_v.20.norm2.bias','blocks_v.20.mlp.fc1.weight','blocks_v.20.mlp.fc1.bias','blocks_v.20.mlp.fc2.weight','blocks_v.20.mlp.fc2.bias',
'blocks_v.21.norm1.weight','blocks_v.21.norm1.bias','blocks_v.21.attn.qkv.weight','blocks_v.21.attn.qkv.bias','blocks_v.21.attn.proj.weight','blocks_v.21.attn.proj.bias','blocks_v.21.norm2.weight','blocks_v.21.norm2.bias','blocks_v.21.mlp.fc1.weight','blocks_v.21.mlp.fc1.bias','blocks_v.21.mlp.fc2.weight','blocks_v.21.mlp.fc2.bias',
'blocks_v.22.norm1.weight','blocks_v.22.norm1.bias','blocks_v.22.attn.qkv.weight','blocks_v.22.attn.qkv.bias','blocks_v.22.attn.proj.weight','blocks_v.22.attn.proj.bias','blocks_v.22.norm2.weight','blocks_v.22.norm2.bias','blocks_v.22.mlp.fc1.weight','blocks_v.22.mlp.fc1.bias','blocks_v.22.mlp.fc2.weight','blocks_v.22.mlp.fc2.bias',
'blocks_v.23.norm1.weight','blocks_v.23.norm1.bias','blocks_v.23.attn.qkv.weight','blocks_v.23.attn.qkv.bias','blocks_v.23.attn.proj.weight','blocks_v.23.attn.proj.bias','blocks_v.23.norm2.weight','blocks_v.23.norm2.bias','blocks_v.23.mlp.fc1.weight','blocks_v.23.mlp.fc1.bias','blocks_v.23.mlp.fc2.weight','blocks_v.23.mlp.fc2.bias',
'patch_embed.proj.weight','patch_embed.proj.bias', 'pos_emb',
'blocks.0.norm1.weight','blocks.0.norm1.bias','blocks.0.attn.qkv.weight','blocks.0.attn.qkv.bias','blocks.0.attn.proj.weight','blocks.0.attn.proj.bias','blocks.0.norm2.weight','blocks.0.norm2.bias','blocks.0.mlp.fc1.weight','blocks.0.mlp.fc1.bias','blocks.0.mlp.fc2.weight','blocks.0.mlp.fc2.bias',
'blocks.1.norm1.weight','blocks.1.norm1.bias','blocks.1.attn.qkv.weight','blocks.1.attn.qkv.bias','blocks.1.attn.proj.weight','blocks.1.attn.proj.bias','blocks.1.norm2.weight','blocks.1.norm2.bias','blocks.1.mlp.fc1.weight','blocks.1.mlp.fc1.bias','blocks.1.mlp.fc2.weight','blocks.1.mlp.fc2.bias',
'blocks.2.norm1.weight','blocks.2.norm1.bias','blocks.2.attn.qkv.weight','blocks.2.attn.qkv.bias','blocks.2.attn.proj.weight','blocks.2.attn.proj.bias','blocks.2.norm2.weight','blocks.2.norm2.bias','blocks.2.mlp.fc1.weight','blocks.2.mlp.fc1.bias','blocks.2.mlp.fc2.weight','blocks.2.mlp.fc2.bias',
'blocks.3.norm1.weight','blocks.3.norm1.bias','blocks.3.attn.qkv.weight','blocks.3.attn.qkv.bias','blocks.3.attn.proj.weight','blocks.3.attn.proj.bias','blocks.3.norm2.weight','blocks.3.norm2.bias','blocks.3.mlp.fc1.weight','blocks.3.mlp.fc1.bias','blocks.3.mlp.fc2.weight','blocks.3.mlp.fc2.bias',
'blocks.4.norm1.weight','blocks.4.norm1.bias','blocks.4.attn.qkv.weight','blocks.4.attn.qkv.bias','blocks.4.attn.proj.weight','blocks.4.attn.proj.bias','blocks.4.norm2.weight','blocks.4.norm2.bias','blocks.4.mlp.fc1.weight','blocks.4.mlp.fc1.bias','blocks.4.mlp.fc2.weight','blocks.4.mlp.fc2.bias',
'blocks.5.norm1.weight','blocks.5.norm1.bias','blocks.5.attn.qkv.weight','blocks.5.attn.qkv.bias','blocks.5.attn.proj.weight','blocks.5.attn.proj.bias','blocks.5.norm2.weight','blocks.5.norm2.bias','blocks.5.mlp.fc1.weight','blocks.5.mlp.fc1.bias','blocks.5.mlp.fc2.weight','blocks.5.mlp.fc2.bias',
'blocks.6.norm1.weight','blocks.6.norm1.bias','blocks.6.attn.qkv.weight','blocks.6.attn.qkv.bias','blocks.6.attn.proj.weight','blocks.6.attn.proj.bias','blocks.6.norm2.weight','blocks.6.norm2.bias','blocks.6.mlp.fc1.weight','blocks.6.mlp.fc1.bias','blocks.6.mlp.fc2.weight','blocks.6.mlp.fc2.bias',
'blocks.7.norm1.weight','blocks.7.norm1.bias','blocks.7.attn.qkv.weight','blocks.7.attn.qkv.bias','blocks.7.attn.proj.weight','blocks.7.attn.proj.bias','blocks.7.norm2.weight','blocks.7.norm2.bias','blocks.7.mlp.fc1.weight','blocks.7.mlp.fc1.bias','blocks.7.mlp.fc2.weight','blocks.7.mlp.fc2.bias',
'blocks.8.norm1.weight','blocks.8.norm1.bias','blocks.8.attn.qkv.weight','blocks.8.attn.qkv.bias','blocks.8.attn.proj.weight','blocks.8.attn.proj.bias','blocks.8.norm2.weight','blocks.8.norm2.bias','blocks.8.mlp.fc1.weight','blocks.8.mlp.fc1.bias','blocks.8.mlp.fc2.weight','blocks.8.mlp.fc2.bias',
'blocks.9.norm1.weight','blocks.9.norm1.bias','blocks.9.attn.qkv.weight','blocks.9.attn.qkv.bias','blocks.9.attn.proj.weight','blocks.9.attn.proj.bias','blocks.9.norm2.weight','blocks.9.norm2.bias','blocks.9.mlp.fc1.weight','blocks.9.mlp.fc1.bias','blocks.9.mlp.fc2.weight','blocks.9.mlp.fc2.bias',
'blocks.10.norm1.weight','blocks.10.norm1.bias','blocks.10.attn.qkv.weight','blocks.10.attn.qkv.bias','blocks.10.attn.proj.weight','blocks.10.attn.proj.bias','blocks.10.norm2.weight','blocks.10.norm2.bias','blocks.10.mlp.fc1.weight','blocks.10.mlp.fc1.bias','blocks.10.mlp.fc2.weight','blocks.10.mlp.fc2.bias',
'blocks.11.norm1.weight','blocks.11.norm1.bias','blocks.11.attn.qkv.weight','blocks.11.attn.qkv.bias','blocks.11.attn.proj.weight','blocks.11.attn.proj.bias','blocks.11.norm2.weight','blocks.11.norm2.bias','blocks.11.mlp.fc1.weight','blocks.11.mlp.fc1.bias','blocks.11.mlp.fc2.weight','blocks.11.mlp.fc2.bias'
'blocks.12.norm1.weight','blocks.12.norm1.bias','blocks.12.attn.qkv.weight','blocks.12.attn.qkv.bias','blocks.12.attn.proj.weight','blocks.12.attn.proj.bias','blocks.12.norm2.weight','blocks.12.norm2.bias','blocks.12.mlp.fc1.weight','blocks.12.mlp.fc1.bias','blocks.12.mlp.fc2.weight','blocks.12.mlp.fc2.bias',
'blocks.13.norm1.weight','blocks.13.norm1.bias','blocks.13.attn.qkv.weight','blocks.13.attn.qkv.bias','blocks.13.attn.proj.weight','blocks.13.attn.proj.bias','blocks.13.norm2.weight','blocks.13.norm2.bias','blocks.13.mlp.fc1.weight','blocks.13.mlp.fc1.bias','blocks.13.mlp.fc2.weight','blocks.13.mlp.fc2.bias',
'blocks.14.norm1.weight','blocks.14.norm1.bias','blocks.14.attn.qkv.weight','blocks.14.attn.qkv.bias','blocks.14.attn.proj.weight','blocks.14.attn.proj.bias','blocks.14.norm2.weight','blocks.14.norm2.bias','blocks.14.mlp.fc1.weight','blocks.14.mlp.fc1.bias','blocks.14.mlp.fc2.weight','blocks.14.mlp.fc2.bias',
'blocks.15.norm1.weight','blocks.15.norm1.bias','blocks.15.attn.qkv.weight','blocks.15.attn.qkv.bias','blocks.15.attn.proj.weight','blocks.15.attn.proj.bias','blocks.15.norm2.weight','blocks.15.norm2.bias','blocks.15.mlp.fc1.weight','blocks.15.mlp.fc1.bias','blocks.15.mlp.fc2.weight','blocks.15.mlp.fc2.bias',
'blocks.16.norm1.weight','blocks.16.norm1.bias','blocks.16.attn.qkv.weight','blocks.16.attn.qkv.bias','blocks.16.attn.proj.weight','blocks.16.attn.proj.bias','blocks.16.norm2.weight','blocks.16.norm2.bias','blocks.16.mlp.fc1.weight','blocks.16.mlp.fc1.bias','blocks.16.mlp.fc2.weight','blocks.16.mlp.fc2.bias',
'blocks.17.norm1.weight','blocks.17.norm1.bias','blocks.17.attn.qkv.weight','blocks.17.attn.qkv.bias','blocks.17.attn.proj.weight','blocks.17.attn.proj.bias','blocks.17.norm2.weight','blocks.17.norm2.bias','blocks.17.mlp.fc1.weight','blocks.17.mlp.fc1.bias','blocks.17.mlp.fc2.weight','blocks.17.mlp.fc2.bias',
'blocks.18.norm1.weight','blocks.18.norm1.bias','blocks.18.attn.qkv.weight','blocks.18.attn.qkv.bias','blocks.18.attn.proj.weight','blocks.18.attn.proj.bias','blocks.18.norm2.weight','blocks.18.norm2.bias','blocks.18.mlp.fc1.weight','blocks.18.mlp.fc1.bias','blocks.18.mlp.fc2.weight','blocks.18.mlp.fc2.bias',
'blocks.19.norm1.weight','blocks.19.norm1.bias','blocks.19.attn.qkv.weight','blocks.19.attn.qkv.bias','blocks.19.attn.proj.weight','blocks.19.attn.proj.bias','blocks.19.norm2.weight','blocks.19.norm2.bias','blocks.19.mlp.fc1.weight','blocks.19.mlp.fc1.bias','blocks.19.mlp.fc2.weight','blocks.19.mlp.fc2.bias',
'blocks.20.norm1.weight','blocks.20.norm1.bias','blocks.20.attn.qkv.weight','blocks.20.attn.qkv.bias','blocks.20.attn.proj.weight','blocks.20.attn.proj.bias','blocks.20.norm2.weight','blocks.20.norm2.bias','blocks.20.mlp.fc1.weight','blocks.20.mlp.fc1.bias','blocks.20.mlp.fc2.weight','blocks.20.mlp.fc2.bias',
'blocks.21.norm1.weight','blocks.21.norm1.bias','blocks.21.attn.qkv.weight','blocks.21.attn.qkv.bias','blocks.21.attn.proj.weight','blocks.21.attn.proj.bias','blocks.21.norm2.weight','blocks.21.norm2.bias','blocks.21.mlp.fc1.weight','blocks.21.mlp.fc1.bias','blocks.21.mlp.fc2.weight','blocks.21.mlp.fc2.bias',
'blocks.22.norm1.weight','blocks.22.norm1.bias','blocks.22.attn.qkv.weight','blocks.22.attn.qkv.bias','blocks.22.attn.proj.weight','blocks.22.attn.proj.bias','blocks.22.norm2.weight','blocks.22.norm2.bias','blocks.22.mlp.fc1.weight','blocks.22.mlp.fc1.bias','blocks.22.mlp.fc2.weight','blocks.22.mlp.fc2.bias',
'blocks.23.norm1.weight','blocks.23.norm1.bias','blocks.23.attn.qkv.weight','blocks.23.attn.qkv.bias','blocks.23.attn.proj.weight','blocks.23.attn.proj.bias','blocks.23.norm2.weight','blocks.23.norm2.bias','blocks.23.mlp.fc1.weight','blocks.23.mlp.fc1.bias','blocks.23.mlp.fc2.weight','blocks.23.mlp.fc2.bias',
]


def param_groups_lrd2(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    param_group_names_v = {}
    param_groups_v = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        #if '_v' in n:
        #    print(n)
        if n in vision_list:
            #print(n)
            if group_name not in param_group_names_v:
                this_scale = layer_scales[layer_id]

                param_group_names_v[group_name] = {
                    "lr_scale": 0.5*this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups_v[group_name] = {
                    "lr_scale": 0.5*this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }            
            param_group_names_v[group_name]["params"].append(n)
            param_groups_v[group_name]["params"].append(p)
        else:
            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],               
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],                
                }
            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

    #print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    #print("parameter groups2: \n%s" % json.dumps(param_group_names_v, indent=2))

    #return param_groups.values(), param_groups_v.values()
    return list(param_groups.values()), list(param_groups_v.values())


def param_groups_lrd_av_slow(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    param_group_names_v = {}
    param_groups_v = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        #if '_v' in n:
        #    print(n)
        if n in audio_video_list:
            #print(n)
            if group_name not in param_group_names_v:
                this_scale = layer_scales[layer_id]

                param_group_names_v[group_name] = {
                    "lr_scale": 0.1*this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups_v[group_name] = {
                    "lr_scale": 0.1*this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }            
            param_group_names_v[group_name]["params"].append(n)
            param_groups_v[group_name]["params"].append(p)
        else:
            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],               
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],                
                }
            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

    #print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    #print("parameter groups2: \n%s" % json.dumps(param_group_names_v, indent=2))

    #return param_groups.values(), param_groups_v.values()
    return list(param_groups.values()), list(param_groups_v.values())



def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed', 'cls_token_v', 'pos_embed_v']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('patch_embed_v'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
