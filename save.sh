# for model in "avbert" "replai" "mavil_base" "avhubert_ft_lrs3_433" "hubert" "avhubert_large_lrs3" "mavil_local -k mavil_as_pt_ft_a+v.pth"
# do
# source /home/rogertseng/.bashrc
# cd /home/rogertseng/audiovisual-benchmark
# conda activate av

model=$1
mkdir -p "features/${model}";
python run_downstream.py -m train \
    -c "downstream_tasks/av_asr/config_save.yaml" \
    -d av_asr \
    -u $model \
    -p "result/store/${model}" \
    --pooled_features_path "features/${model}" # 2>&1 | tee -a "features/${model}/out.log"
# done