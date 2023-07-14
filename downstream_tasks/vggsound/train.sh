cd ../../
# python3 run_downstream.py -m train -n avbert_vggsound_fusion_lr1E03 -u avbert -d vggsound -s fusion_feats
# python3 run_downstream.py -m train -e ./result/downstream/avbert_vggsound_fusion_lr1E03/states-50.ckpt

# python3 run_downstream.py -m train -n avhubert_vggsound_fusion_lr1E04 -u avhubert -d vggsound -s fusion_feats


# python3 run_downstream.py -m train -n avhubert_ft_lrs3_433_vggsound_fusion_lr1E02 -u avhubert_ft_lrs3_433 -d vggsound -s fusion_feats --pooled_features_path /work/u2707828/features/

# python3 run_downstream.py -m train -n avhubert_vggsound_video_lr1E05 -u avhubert -d vggsound -s video_feats --pooled_features_path /work/u2707828/features/


# python3 run_downstream.py -m train -n mavil_vggsound_fusion_lr1E02 -u mavil_base -k /work/u2707828/mavil_as_pt_ft_a+v.pth -d vggsound -s fusion_feats --pooled_features_path /work/u2707828/features/


# python3 run_downstream.py -m train -n testhubert -u hubert -d vggsound -s audio_feats --pooled_features_path /work/u2707828/features/

# python3 run_downstream.py -m train -n testavhubert -u avhubert -d vggsound -s audio_feats --pooled_features_path /work/u2707828/features/

# python3 run_downstream.py -m train -n testavhubert_large_lrs3 -u avhubert_large_lrs3 -d vggsound -s fusion_feats --pooled_features_path /work/u2707828/features/

