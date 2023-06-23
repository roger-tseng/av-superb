cd ../../
# python3 run_downstream.py -m train -n avbert_vggsound_fusion_lr1E03 -u avbert -d vggsound -s fusion_feats
# python3 run_downstream.py -m train -e ./result/downstream/avbert_vggsound_fusion_lr1E03/states-50.ckpt

# python3 run_downstream.py -m train -n avhubert_vggsound_fusion_lr1E04 -u avhubert -d vggsound -s fusion_feats
python3 run_downstream.py -m train -n avhubert_vggsound_video_lr1E03 -u avhubert -d vggsound -s video_feats
