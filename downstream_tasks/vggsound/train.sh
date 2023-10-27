cd ../../
python3 run_downstream.py -m train -n [name] -u [upstream model] -d vggsound -s [audio_feats | video_feats | fusion_feats] --pooled_features_path [feature directory path]

