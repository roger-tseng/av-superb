AS_PATH="/work/b07901163/av/AudioSet-20K"
if [ ! -d "$AS_PATH" ]; then
    echo "$AS_PATH does not exist."
else
    echo "changing eval labels to int labels"
    python3 downstream_tasks/audioset/audioset_preprocess/change_label.py \
        --csv ${AS_PATH}/eval_segments.csv \
        --class_csv ${AS_PATH}/class_labels_indices.csv \
        --output_dir downstream_tasks/audioset/audioset_preprocess/csv  \
        --output_filename eval_audioset_int_labels.csv

    echo "changing train labels to int labels"
    python3 downstream_tasks/audioset/audioset_preprocess/change_label.py \
        --csv ${AS_PATH}/balanced_train_segments.csv \
        --class_csv ${AS_PATH}/class_labels_indices.csv \
        --output_dir downstream_tasks/audioset/audioset_preprocess/csv  \
        --output_filename train_audioset_int_labels.csv

    echo "checking existing training datasets"
    python3 downstream_tasks/audioset/audioset_preprocess/check_dataset.py \
        --split train \
        --csv downstream_tasks/audioset/audioset_preprocess/csv/train_audioset_int_labels.csv \
        --dest downstream_tasks/audioset/audioset_preprocess/csv \
        --data ${AS_PATH}/balanced_train/video_mp4_288p

    echo "checking existing testing datasets"
    python3 downstream_tasks/audioset/audioset_preprocess/check_dataset.py \
        --split test \
        --csv downstream_tasks/audioset/audioset_preprocess/csv/eval_audioset_int_labels.csv \
        --dest downstream_tasks/audioset/audioset_preprocess/csv \
        --data ${AS_PATH}/eval/video_mp4_288p
fi