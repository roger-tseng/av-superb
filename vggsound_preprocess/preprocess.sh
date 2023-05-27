rm -r ../split/
mkdir ../split/
mkdir ../split/ref/
mkdir ../split/split/

echo "Get vggsound dataset csv file"
wget -O ../split/ref/vggsound.csv https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv

echo "Changing labels to int labels"
python3 change_label.py --csv ../split/ref/vggsound.csv --output_dir ../split/ref --output_filename vggsound_int_labels.csv

echo "Checking existing video datasets"
python3 check_dataset.py --csv ../split/ref/vggsound_int_labels.csv --dest ../split/split --data /work/u2707828/VGGSound/VGGSound/ --reduce_class 310
