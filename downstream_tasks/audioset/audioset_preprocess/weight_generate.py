# ref:https://github.com/YuanGongND/psla/blob/main/src/gen_weight_file.py
import argparse
import csv

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True, help="Path of original csv")
parser.add_argument(
    "--dest", type=str, required=True, help="Path for output csv directory"
)
parser.add_argument("--split", type=str, required=True, help="Train or test")

if __name__ == "__main__":
    args = parser.parse_args()

    num_class = 527
    label_count = np.zeros(num_class)

    datas = list([])
    with open(args.csv, "r") as csvfile:
        datas = list(csv.reader(csvfile, delimiter=","))

    for data in datas:
        labels = data[3:]
        for label in labels:
            label_count[int(label)] += 1
    # the reason not using 1 is to avoid underflow for majority classes, add small value to avoid underflow
    label_weight = 1000.0 / (label_count + 0.01)
    sample_weight = np.zeros(len(datas))

    for i, sample in enumerate(datas):
        sample_labels = sample[3:]
        max_label_weight = 0
        for label in sample_labels:
            if label_weight[int(label)] > max_label_weight:
                max_label_weight = label_weight[int(label)]
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            # sample_weight[i] += label_weight[int(label)]
        sample_weight[i] = max_label_weight
    dest_folder = args.dest
    if dest_folder[-1] != "/":
        dest_folder += "/"
    np.savetxt(
        dest_folder + f"audioset_{args.split}_weight.csv", sample_weight, delimiter=","
    )
