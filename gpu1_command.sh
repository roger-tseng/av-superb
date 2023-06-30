# Running on gpu 5 actually!
CUDA_VISIBLE_DEVICES=5 python3 run_downstream.py -m train -n gpu1_test_fusion -u avhubert -d av_asr -s fusion_feats -c cfg_gpu1_test.yaml
CUDA_VISIBLE_DEVICES=5 python3 run_downstream.py -m train -n gpu1_test_audio -u avhubert -d av_asr -s audio_feats -c cfg_gpu1_test.yaml
CUDA_VISIBLE_DEVICES=5 python3 run_downstream.py -m train -n gpu1_test_video -u avhubert -d av_asr -s video_feats -c cfg_gpu1_test.yaml
