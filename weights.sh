for model in "mavil_base -s audio_seq_feats" "avhubert" "avbert" "replai" "avhubert_ft_lrs3_433" "hubert" "avhubert_large_lrs3" "mavil_local -k /home/rogertseng/audiovisual-benchmark/mavil_as_pt_ft_a+v.pth -s audio_seq_feats"
do
    python weights.py $model # ls "/home/rogertseng/audiovisual-benchmark/features/${model}"
done