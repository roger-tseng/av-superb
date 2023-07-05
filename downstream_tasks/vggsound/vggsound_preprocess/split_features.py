import os
import csv 
import argparse

def addArgument(parser, name, required, help):
    parser.add_argument(name, str, required = required, help = help)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split", type = str, required = True, help = "Path of train split csv")
    parser.add_argument("--output_dir", type = str, required = True, help = "Path of output directory")
    parser.add_argument("--feature_dir", type = str, required = True, help = "Path of features directory")

    args = parser.parse_args()

    dest_folder = args.output_dir
    if dest_folder[-1] != "/":  dest_folder += "/"

    with open(args.train_split,"r") as csvfile:
        train_datas = list(csv.reader(csvfile,delimiter=","))    

    feature_train, no_feature_train = list(), list()

    def getFeaturePath(data):
        start_time = str(int(data[1]))
        filename = "_".join([data[0], (6 - len(start_time)) * "0" + start_time + ".pt"])
        feature_path = f"{args.feature_dir}/{filename}"
        return feature_path

    for train_data in train_datas:
        feature_path = getFeaturePath(train_data)
        if os.path.exists(feature_path):
            feature_train.append(train_data)
        else:
            no_feature_train.append(train_data)

    def writecsv(datas, filepath):
        with open(filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(datas)
        return filepath
    
    writecsv(feature_train, dest_folder + "vggsound_train_saved_features.csv")
    writecsv(no_feature_train, dest_folder + "vggsound_train_not_saved_features.csv")

