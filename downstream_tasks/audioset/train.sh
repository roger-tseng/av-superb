cd ../../


python3 run_downstream.py -m train  -u fbank  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/fbank_connector_mean_linear_audio_-1/ -o "config.optimizer.lr=1.0e-1"
python3 run_downstream.py -m train  -u fbank  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/fbank_connector_mean_linear_audio_-2/ -o "config.optimizer.lr=1.0e-2"
python3 run_downstream.py -m train  -u fbank  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/fbank_connector_mean_linear_audio_-3/ -o "config.optimizer.lr=1.0e-3"
python3 run_downstream.py -m train  -u fbank  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/fbank_connector_mean_linear_audio_-4/ -o "config.optimizer.lr=1.0e-4"
python3 run_downstream.py -m train  -u fbank  -d audioset -s audio_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/fbank_connector_mean_linear_audio_-5/ -o "config.optimizer.lr=1.0e-5"




