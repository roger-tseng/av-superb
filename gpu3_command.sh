# Running on gpu 7 actually!
CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu3_test_fusion -u avhubert -d av_asr -s fusion_feats -c cfg_gpu3_test.yaml
CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu3_test_audio -u avhubert -d av_asr -s audio_feats -c cfg_gpu3_test.yaml
CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu3_test_video -u avhubert -d av_asr -s video_feats -c cfg_gpu3_test.yaml
