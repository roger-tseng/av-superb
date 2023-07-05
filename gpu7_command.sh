CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu7_test_fusion -u avhubert -d av_asr -s fusion_feats -c cfg_gpu7_test.yaml
CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu7_test_audio -u avhubert -d av_asr -s audio_feats -c cfg_gpu7_test.yaml
CUDA_VISIBLE_DEVICES=7 python3 run_downstream.py -m train -n gpu7_test_video -u avhubert -d av_asr -s video_feats -c cfg_gpu7_test.yaml
