# VGG-Sound

```
.
|-- README.md
|-- __init__.py
|-- config.yaml
|-- dataset.py
|-- expert.py
|-- model.py
|-- split
|   |-- ref
|   |   |-- labels.csv
|   |   |-- vggsound.csv
|   |   `-- vggsound_int_labels.csv
|   `-- split
|       |-- vggsound_dev.csv
|       |-- vggsound_test.csv
|       `-- vggsound_train.csv
|-- train.sh
`-- vggsound_preprocess
    |-- change_label.py
    |-- check_dataset.py
    |-- csv_len.py
    |-- preprocess.sh
    `-- split_features.py

4 directories, 18 files
```


## Data Preprocess
The data preprocess codes are stored in `vggsound_preprocess/` and the process includes several parts listed below:
1. Download the csv file `vggsound.csv`
2. Change the labels type from `string` to `int` and create a dictionary file which maps each number to the original label.
3. Check the dataset and filter out the missed data.
4. Finally split the csv file into `train` and `test` according to the official setup.


To preprocess, several steps are needed to be done:
1. Change the `DATA_PATH` value in `vggsound_preprocess/preprocess.sh` to the place data has been stored.
2. Make sure the data name is in the format of `[Youtube ID]_[6-Digit Video Start time].mp4`, otherwise the function `get_file_name` in `vggsound_preprocess/check_dataset.py` should be modified for custom needs.
3. Move into the dictionary `vggsound_preprocess/` and simply run the command `bash preprocess.sh`. The processed csv files would be stored in `split/ref/` and result csv files are in `split/split/`.

## Training
Note that if the data name isn't in the format of `[Youtube ID]_[6-Digit Video Start time].mp4`, codes in `dataset.py` should be changed.
 


