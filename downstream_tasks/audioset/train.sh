cd ../../
python3 run_downstream.py -m train  -u avbert  -d audioset -s video_feats --pooled_features_path /work/u7196393/pooled_features -p /work/u7196393/result/avbert_connector_mean_linear_video_-3/
#-k /home/u7196393/mavil/mavil_as_pt_ft_a+v.pth
