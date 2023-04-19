import argparse
import csv
import os


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",type=str,required=True,help='Path of original csv')
    parser.add_argument("--output_dir",type=str,required=True,help='Path of output directory')
    parser.add_argument("--output_filename",type=str,required=True,help='Output csv file name')

    args = parser.parse_args()

    dest_folder = args.output_dir
    if dest_folder[-1] != "/":  dest_folder += "/"

    with open(args.csv,"r") as csvfile:
        datas = list(csv.reader(csvfile,delimiter=","))
    
    labels = dict([])
    labels_inverse = dict([])

    newdatas = list([])

    for data in datas:
        newdata = data
        if data[2] not in labels:
            labels[data[2]] = len(labels)
            labels_inverse[labels[data[2]]] = data[2]
        newdata[2] = str(labels[data[2]])
        newdatas.append(newdata)


    with open(dest_folder+args.output_filename,"w") as f:
        writer = csv.writer(f)
        writer.writerows(newdatas)

    with open(dest_folder+"labels.csv","w") as f:
        writer = csv.writer(f)
        for i in range(len(labels)):
            writer.writerow([labels_inverse[i],i])
    

