# for model in "avbert" "replai" "mavil_base" "avhubert_ft_lrs3_433" "hubert" "avhubert_large_lrs3" "mavil_local -k /home/rogertseng/audiovisual-benchmark/mavil_as_pt_ft_a+v.pth"
# do
source /home/rogertseng/.bashrc
cd /home/rogertseng/audiovisual-benchmark
conda activate av

for lr in "1.0e-3" "1.0e-4" "1.0e-5" "1.0e-2" "1.0e-1" 
do
    for model in avhubert_audio avhubert_video
    do
        for feats in fusion_feats
        do
            mkdir -p "/home/rogertseng/audiovisual-benchmark/features/${model}";
            python run_downstream.py -m train \
                -c /home/rogertseng/audiovisual-benchmark/downstream_tasks/av_asr/config_layne_best_avbert.yaml \
                -d av_asr \
                -u $model \
                -s $feats \
                -p "/home/rogertseng/audiovisual-benchmark/result/av_asr/${model}/${feats}/${lr}_layne" \
                --pooled_features_path "/home/rogertseng/audiovisual-benchmark/features/${model}" \
                -o "config.optimizer.lr=${lr}" # 2>&1 | tee -a "/home/rogertseng/audiovisual-benchmark/features/${model}/out.log"
        done
    done
done