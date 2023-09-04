cd ../../
#python3 run_downstream.py -m train  -u avhubert_large_lrs3  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_large_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
#python3 run_downstream.py -m train  -u avhubert_large_lrs3  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_large_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"
#fusion of ft
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_-3/ -o "config.optimizer.lr=1.0e-3"
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_-4/ -o "config.optimizer.lr=1.0e-4"
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s fusion_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_-5/ -o "config.optimizer.lr=1.0e-5"
#audio of ft
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_audio_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_audio_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_audio_-3/ -o "config.optimizer.lr=1.0e-3"
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_audio_-4/ -o "config.optimizer.lr=1.0e-4"
#python3 run_downstream.py -m train  -u avhubert_ft_lrs3_433  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avhubert_ft_connector_mean_linear_audio_-5/ -o "config.optimizer.lr=1.0e-5"

