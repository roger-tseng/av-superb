cd ../../
# python3 run_downstream.py -m train -n avbert_vggsound_fusion_lr1E03 -u avbert -d vggsound -s fusion_feats
# python3 run_downstream.py -m train -e ./result/downstream/avbert_vggsound_fusion_lr1E03/states-50.ckpt

# python3 run_downstream.py -m train -n avhubert_vggsound_fusion_lr1E04 -u avhubert -d vggsound -s fusion_feats
# python3 run_downstream.py -m train -n mavilbase_vggsound_video_lr1E05 -u mavil_base -d vggsound -s video_feats --pooled_features_path /work/u2707828/features/


# python3 run_downstream.py -m train -n testreplaitest -u replai -d vggsound -s video_feats --pooled_features_path /work/u2707828/features/

# python3 run_downstream.py -m train -n testmavilbasetest -u mavil_base -d vggsound -s fusion_feats --pooled_features_path /work/u2707828/features/

python3 run_downstream.py -m train -n testavberttest1234582 -u avbert -d vggsound -s fusion_feats --pooled_features_path /work/u2707828/features/
