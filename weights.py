import json
import os
from collections import defaultdict
from glob import glob

import torch
import torch.nn.functional as F
from tqdm import tqdm

paths = {
    # "avhubert_audio": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/avhubert_audio_connector_mean_linear_-3/states-5000.ckpt",
    # },
    # "avhubert_video": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/avhubert_video_connector_mean_linear_-3/states-5000.ckpt",
    # },
    # "avhubert": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/new_avhubert_connector_mean_linear_-3/states-5000.ckpt",
    # },
    # "replai": {
    #     "audio": "NA", # "result/av_asr/replai/audio_feats/1.0e-3_layne/dev-best.ckpt",
    #     "video": "NA", #"result/av_asr/replai/video_feats/1.0e-3_layne/dev-best.ckpt",
    #     "fusion": "NA",
    # },
    "avbert": {
        "audio": "/work/u7196393/result/avbert_connector_mean_linear_audio_-3/states-5000.ckpt",
        "video": "/work/u7196393/result/avbert_connector_mean_linear_video_-3/states-5000.ckpt",
        "fusion": "/work/u7196393/result/avbert_connector_mean_linear_-3/states-5000.ckpt",
    },
    "mavil_base": {
        "audio": "/work/u7196393/result/mavil_base_std_connector_mean_linear_audio_-3/states-5000.ckpt",
        "video": "/work/u7196393/result/mavil_base_std_connector_mean_linear_video_-3/states-5000.ckpt",
        "fusion": "/work/u7196393/result/mavil_base_std_connector_mean_linear_-4/states-5000.ckpt",
    },
    "mavil_local": {
        "audio": "/work/u7196393/result/mavil_local_std_connector_mean_linear_audio_-4/states-5000.ckpt",
        "video": "/work/u7196393/result/mavil_local_std_connector_mean_linear_video_-3/states-5000.ckpt",
        "fusion": "/work/u7196393/result/mavil_local_std_connector_mean_linear_-4/states-5000.ckpt",
    },
    # "hubert": {
    #     "audio": "/work/u7196393/result/hubert_connector_mean_linear_audio_-2/states-5000.ckpt",
    #     "video": "NA",
    #     "fusion": "NA",
    # },
    # "avhubert_ft_lrs3_433": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/avhubert_ft_connector_mean_linear_-2/states-5000.ckpt",
    # },
    # "avhubert_ft_lrs3_433_audio": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-4/states-5000.ckpt",
    # },
    # "avhubert_ft_lrs3_433_video": {
    #     "audio": "NA",
    #     "video": "NA",
    #     "fusion": "/work/u7196393/result/avhubert_ft_video_connector_mean_linear_-3/states-5000.ckpt",
    # },
}

for model, ckpts in paths.items():
    features_path = f"/work/u7196393/pooled_features"

    print(f"Model: {model}")

    os.makedirs("result/weight_analysis", exist_ok=True)
    f = open(f"result/weight_analysis/{model}.json", "w")
    all_save_dict = dict()

    for feat_type, ckpt in ckpts.items():
        all_save_dict[feat_type] = dict()
        save_dict = all_save_dict[feat_type]

        # skip not applicable feature types
        if ckpt == "NA":
            continue

        # Load weighted-sum weights from checkpoint
        ckpt = torch.load(ckpt, map_location="cpu")
        weights = ckpt.get("Featurizer").get("weights")
        try:
            norm_weights = F.softmax(weights, dim=-1).cpu().double().tolist()
        except AttributeError as e:
            raise Exception(
                "This model checkpoint probably does not use weighted-sum"
            ) from e
        save_dict["weights"] = norm_weights

        # Normalize by average L2-norm of features
        model_features_path = f"{features_path}/{model}_{feat_type}"
        print(f"Searching {model_features_path} for {model} features")

        layer_norms = defaultdict(lambda: [])
        for feat in tqdm(
            glob(f"{features_path}/{model}_{feat_type}**/*.pt", recursive=True)[:10000]
        ):
            feature = torch.load(feat)
            assert isinstance(
                feature, list
            ), f"{feat_type} features of {model} may not have used weighted-sum!"
            assert len(feature) == len(
                weights
            ), f"should have {len(weights)} layers, but found {len(weights)} features"

            for i, layer in enumerate(feature):
                layer_norms[f"layer{i}"].append(layer.norm(p=2, dim=0))

        save_dict["layer_norm"] = []
        for i, layer in enumerate(feature):
            avg_layer_norm = torch.hstack(layer_norms[f"layer{i}"]).mean().item()
            save_dict["layer_norm"].append(avg_layer_norm)

        assert len(save_dict["layer_norm"]) == len(save_dict["weights"])
        save_dict[f"norm_weights"] = [
            a * b for a, b in zip(save_dict["layer_norm"], save_dict["weights"])
        ]

    json.dump(all_save_dict, f, indent=4)

    f.close()
