
# Download Data
To obtain more comprehensive data for Kinetics-Sounds, we begin by acquiring the Kinetics 400 dataset and then selecting the classes that belong to Kinetics-Sounds.

You can obtain the Kinetics 400 dataset from:
[https://github.com/cvdfoundation/kinetics-dataset#kinetics-400](https://github.com/cvdfoundation/kinetics-dataset#kinetics-400)

Once you have downloaded the K400 dataset, there should be four directories: `replacement`, `test`, `train`, and `val`, within the `K400` directory.

Finally, you need to add the path settings to the `config.yaml` file. The basic config file can be found in the `example` folder. The `kinetics_root` should point to the `k400` directory. Additionally, `train_data_path.csv`, `val_data_path.csv`, and `test_data_path.csv` have been provided:

```yaml
downstream_expert:
  datarc:
    class_num: 32
    kinetics_root: "path to the K400 directory" 
    train_meta_location: "path to train_data_path.csv"
    val_meta_location: "path to val_data_path.csv"
    test_meta_location: "path to test_data_path.csv"
```
