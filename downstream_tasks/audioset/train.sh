cd ../../
python3 run_downstream.py -m train  -u avhubert_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_audio_connector_mean_linear_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u avhubert_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_audio_connector_mean_linear_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u avhubert_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_audio_connector_mean_linear_-3/ -o "config.optimizer.lr=1.0e-3"
python3 run_downstream.py -m train  -u avhubert_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_audio_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
python3 run_downstream.py -m train  -u avhubert_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_audio_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"


python3 run_downstream.py -m train  -u avhubert_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_video_connector_mean_linear_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u avhubert_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_video_connector_mean_linear_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u avhubert_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_video_connector_mean_linear_-3/ -o "config.optimizer.lr=1.0e-3"
python3 run_downstream.py -m train  -u avhubert_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_video_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
python3 run_downstream.py -m train  -u avhubert_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_video_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"


python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-3/ -o "config.optimizer.lr=1.0e-3"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_audio  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_audio_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"


python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_video_connector_mean_linear_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_video_connector_mean_linear_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_video_connector_mean_linear_-3/ -o "config.optimizer.lr=1.0e-3"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_video_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433_video  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_video_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"


