echo "changing eval labels to int labels"
python3 change_label.py --csv /work/u3933430/testAudioset/csv/eval_segments.csv --class_csv /work/u3933430/testAudioset/csv/class_labels_indices.csv --output_dir /work/u3933430/testAudioset/csv/  --output_filename eval_audioset_int_labels.csv

echo "changing train labels to int labels"
python3 change_label.py --csv /work/u3933430/testAudioset/csv/balanced_train_segments.csv --class_csv /work/u3933430/testAudioset/csv/class_labels_indices.csv --output_dir /work/u3933430/testAudioset/csv/  --output_filename train_audioset_int_labels.csv

echo "checking existing training datasets"
python3 check_dataset.py --split train --csv /work/u3933430/testAudioset/csv/train_audioset_int_labels.csv --dest /work/u3933430/testAudioset/csv/ --data /work/u3933430/testAudioset/data/balanced_train/video_mp4_288p

echo "checking existing testing datasets"
python3 check_dataset.py --split test --csv /work/u3933430/testAudioset/csv/eval_audioset_int_labels.csv --dest /work/u3933430/testAudioset/csv/ --data /work/u3933430/testAudioset/data/eval/video_mp4_288p