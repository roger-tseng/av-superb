import argparse
import csv
import os
import random


def getFilename(l):
    filename = "_".join([l[0] + ".mp4"])
    return filename


def writecsv(datas, filepath):
    with open(filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerows(datas)
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="Train or test")
    parser.add_argument("--csv", type=str, required=True, help="Path of original csv")
    parser.add_argument(
        "--dest", type=str, required=True, help="Path for output csv directory"
    )
    parser.add_argument("--data", type=str, required=True, help="Path of audio dataset")

    args = parser.parse_args()

    assert os.path.isfile(args.csv), '"' + args.csv + '" not found'
    assert os.path.isdir(args.data), '"' + args.data + '" not found'
    assert os.path.isdir(args.dest), '"' + args.dest + '" not found'

    data_folder = args.data
    if data_folder[-1] != "/":
        data_folder += "/"

    dest_folder = args.dest
    if dest_folder[-1] != "/":
        dest_folder += "/"

    datas = list([])

    train = list([])
    valid = list([])
    test = list([])

    with open(args.csv, "r") as csvfile:
        datas = list(csv.reader(csvfile, delimiter=","))

    dictionary = dict([])
    for data in datas:
        dictionary[getFilename(data)] = data

    lsla = [
        i.split()
        for i in os.popen("ls  %s -la" % (data_folder)).read().split("\n")[1:-1]
    ]

    for file in lsla:
        filename = file[8]
        filesize = int(file[4])
        if filename == ".":
            continue
        if filename == "..":
            continue

        data = dictionary[filename]
        train.append(data)
    # print(len(lsla), len(train), len(os.listdir(data_folder)))
    # print(os.listdir(data_folder)[0])
    print("starting split valid and train")
    random.seed(324)
    random.shuffle(train)
    if args.split == "train":
        train_len = int(1.0 * len(train))
        valid_len = int(0 * len(train))
        test_len = int(0 * len(train))
        test = train[valid_len + train_len :]
        valid = train[train_len : train_len + valid_len]
        train = train[:train_len]
        print("writing train/valid csv")
        writecsv(train, dest_folder + "audioset_train.csv")
        writecsv(valid, dest_folder + "audioset_dev.csv")
    elif args.split == "test":
        train_len = int(0 * len(train))
        valid_len = int(0 * len(train))
        test_len = int(1 * len(train))
        test = train[valid_len + train_len :]
        valid = train[train_len : train_len + valid_len]
        train = train[:train_len]
        print("writing test csv")
        writecsv(test, dest_folder + "audioset_test.csv")
