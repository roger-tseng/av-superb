from collections import defaultdict
from glob import glob
import json
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F

paths = {
    "avhubert_audio": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_audio_vggsound_lr1E03/states-5000.ckpt", 
    },
    "avhubert_video": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_video_vggsound_lr1E03/states-5000.ckpt", 
    },
    "avhubert": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_vggsound_fusion_lr1E03/states-5000.ckpt", 
    },
    "replai": {
        "audio": "NA", # result/downstream/replai_vggsound_audio_lr1E02/states-5000.ckpt
        "video": "NA", # result/downstream/replai_vggsound_video_lr1E03/states-5000.ckpt  
        "fusion": "NA", 
    },
    "avbert": {
        "audio": "result/downstream/avbert_vggsound_audio_lr1E03/states-5000.ckpt", 
        "video": "result/downstream/avbert_vggsound_video_lr1E03/states-5000.ckpt", 
        "fusion": "result/downstream/avbert_vggsound_fusion_lr1E03/states-5000.ckpt", 
    },
    "mavil_base": {
        "audio": "result/downstream/mavilbase_vggsound_audio_lr1E03/states-5000.ckpt", 
        "video": "result/downstream/mavilbase_vggsound_video_lr1E04/states-5000.ckpt", 
        "fusion": "result/downstream/mavilbase_vggsound_fusion_lr1E04/states-5000.ckpt", 
    },
    "mavil_local": {
        "audio": "result/downstream/mavil_vggsound_audio_lr1E03/states-5000.ckpt", 
        "video": "result/downstream/mavil_vggsound_video_lr1E04/states-5000.ckpt", 
        "fusion": "result/downstream/mavil_vggsound_fusion_lr1E04/states-5000.ckpt", 
    },
    "hubert": {
        "audio": "result/downstream/hubert_vggsound_audio_lr1E03/states-5000.ckpt", 
        "video": "NA", 
        "fusion": "NA", 
    },
    "avhubert_ft_lrs3_433": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_ft_lrs3_433_vggsound_fusion_lr1E03/states-5000.ckpt", 
    },
    "avhubert_ft_lrs3_433_audio": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_ft_lrs3_433_audio_vggsound_lr1E03/states-5000.ckpt", 
    },
    "avhubert_ft_lrs3_433_video": {
        "audio": "NA", 
        "video": "NA", 
        "fusion": "result/downstream/avhubert_ft_lrs3_433_video_vggsound_lr1E03/states-5000.ckpt", 
    },
}

for model, ckpts in paths.items():

    features_path = f"/work/u2707828/features/"

    print(f"Model: {model}")

    os.makedirs('result/weight_analysis', exist_ok=True)
    f = open(f'result/weight_analysis/{model}.json', 'w')
    all_save_dict = dict()

    for feat_type, ckpt in ckpts.items():

        all_save_dict[feat_type] = dict()
        save_dict = all_save_dict[feat_type]

        # skip not applicable feature types
        if ckpt == 'NA':
            continue

        # Load weighted-sum weights from checkpoint 
        ckpt = torch.load(ckpt, map_location='cpu')
        weights = ckpt.get('Featurizer').get('weights')
        try:
            norm_weights = F.softmax(weights, dim=-1).cpu().double().tolist()
        except AttributeError as e:
            raise Exception('This model checkpoint probably does not use weighted-sum') from e
        save_dict['weights'] = norm_weights

        # Normalize by average L2-norm of features
        model_features_path = f"{features_path}/{model}_{feat_type}"
        print(f"Searching {model_features_path} for {model} features")
        
        layer_norms = defaultdict(lambda: [])
        for feat in tqdm(glob(f"{features_path}/{model}_{feat_type}**/*.pt", recursive=True)[:10000]):
            feature = torch.load(feat)
            assert isinstance(feature, list), f"{feat_type} features of {model} may not have used weighted-sum!"
            assert len(feature) == len(weights), f"should have {len(weights)} layers, but found {len(feature)} features"
            
            for i, layer in enumerate(feature):
                layer_norms[f'layer{i}'].append(layer.norm(p=2,dim=0))

        save_dict['layer_norm'] = []
        for i, layer in enumerate(feature):
            avg_layer_norm = torch.hstack(layer_norms[f'layer{i}']).mean().item()
            save_dict['layer_norm'].append(avg_layer_norm)

        assert len(save_dict['layer_norm']) == len(save_dict['weights'])
        save_dict[f'norm_weights'] = [a*b for a,b in zip(save_dict['layer_norm'], save_dict['weights'])]

    json.dump(all_save_dict, f, indent = 4)

    f.close()
