echo "changing labels to int labels"
python3 change_label.py --csv /work/u3933430/testAudioset/csv/eval_segments.csv --class_csv /work/u3933430/testAudioset/csv/class_labels_indices.csv --output_dir /work/u3933430/testAudioset/csv/  --output_filename audioset_int_labels.csv

echo "checking existing audio datasets"
python3 check_dataset.py --csv /work/u3933430/testAudioset/csv/audioset_int_labels.csv --dest /work/u3933430/testAudioset/csv/ --data /work/u3933430/testAudioset/data/eval/video_mp4_288p
