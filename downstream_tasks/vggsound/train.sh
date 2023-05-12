cd ../../
# python3 run_downstream.py -m train -n vggsound -u customized_upstream -d vggsound -s fusion_feats
python3 run_downstream.py -m train -n vggsound_avhubert0512 -u avhubert -d vggsound -s fusion_feats
# python3 run_downstream.py -m train -n vggsound_avhubert0512_video_feats -u avhubert -d vggsound -s video_feats
# python3 run_downstream.py -m train -n vggsound_replai0511 -u replai -d vggsound -s video_feats


