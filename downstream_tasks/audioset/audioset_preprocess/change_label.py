import argparse
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path of original csv")
    parser.add_argument(
        "--class_csv", type=str, required=True, help="Path of class_labels_indices csv"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path of output directory"
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Output csv file name"
    )

    args = parser.parse_args()

    dest_folder = args.output_dir
    if dest_folder[-1] != "/":
        dest_folder += "/"

    with open(args.csv, "r") as csvfile:
        datas = list(csv.reader(csvfile, delimiter=","))

    with open(args.class_csv, "r") as csvfile:
        class_labels = list(csv.reader(csvfile, delimiter=","))

    mid_to_index = dict([])

    for line in class_labels[1:]:
        index = line[0]
        mid = line[1]
        # print(mid)
        mid_to_index[mid] = index

    # labels = dict([])
    # labels_inverse = dict([])
    # print(mid_to_index['/m/068hy'])

    newdatas = list([])

    for data in datas[3:]:
        newdata = data
        # if data[2] not in labels:
        #     labels[data[2]] = len(labels)
        #     labels_inverse[labels[data[2]]] = data[2]
        midlist = data[3:]
        midlist[0] = midlist[0][2:]
        # print(midlist[0])
        midlist[-1] = midlist[-1][:-1]
        # print(midlist[-1])
        cnt = 0
        for mid in midlist:
            newdata[3 + cnt] = str(mid_to_index[mid])
            cnt += 1
        # newdata[2] = str(labels[data[2]])
        newdatas.append(newdata)

    with open(dest_folder + args.output_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(newdatas)

    # with open(dest_folder+"labels.csv","w") as f:
    #     writer = csv.writer(f)
    #     for i in range(len(labels)):
    #         writer.writerow([labels_inverse[i],i])
