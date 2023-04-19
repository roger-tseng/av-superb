echo -n "Reduce class num (total number of classes = 310): "
read class_num


echo "changing labels to int labels"
python3 change_label.py --csv ./csv/vggsound.csv --output_dir ./csv --output_filename vggsound_int_labels.csv

echo "checking existing audio datasets"
python3 check_dataset.py --csv ./csv/vggsound_int_labels.csv --dest ../vggsound --data /work/u3933430/vggaudiosetdl/vggsound/data/vggsound/audio --reduce_class $class_num
