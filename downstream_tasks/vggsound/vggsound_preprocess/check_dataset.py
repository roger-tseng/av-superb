import argparse
import csv
import random
import os

# Get the video file name from raw data [Youtube ID, Start time, Label, split]
def getFilename(l):
    start_time = str(int(l[1]))
    filename = "_".join([l[0], (6 - len(start_time)) * "0" + start_time + ".mp4"])
    return filename

# Write train/dev/test csv file back
def writecsv(datas, filepath):
    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(datas)
    return filepath



if __name__ == '__main__':

    num_classes = 310

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",type=str,required=True,help='Path of original csv')
    parser.add_argument("--dest",type=str,required=True,help='Path for output csv directory')
    parser.add_argument("--data",type=str,required=True,help='Path of video dataset')
    parser.add_argument("--reduce_class",type=str,help='Reduced num of class')


    args = parser.parse_args()

    chosen_class = num_classes if not args.reduce_class else int(args.reduce_class)

    print("num of chosen classes = {}".format(chosen_class))

    assert os.path.isfile(args.csv), '\"' + args.csv + '\" not found'
    assert os.path.isdir(args.data), '\"' + args.data + '\" not found'
    assert os.path.isdir(args.dest), '\"' + args.dest + '\" not found'

    data_folder, dest_folder = args.data, args.dest
    if data_folder[-1] != "/":  data_folder += "/"
    if dest_folder[-1] != "/":  dest_folder += "/"

    datas, train, valid, test = list(), list(), list(), list()


    with open(args.csv,"r") as csvfile:	datas = list(csv.reader(csvfile,delimiter=","))

    dictionary = dict([])
    for data in datas:	dictionary[getFilename(data)] = data


    lsla = os.listdir(data_folder)
    print("There're {} files in video directory".format(len(lsla)))


    for filename in lsla:
        test.append(dictionary[filename]) if dictionary[filename][-1]=='test' else train.append(dictionary[filename])


    print("train/valid/test csv length = {}/{}/{}".format(len(train),len(valid),len(test)))


    print("Writing train/valid/test csv")
    writecsv(train, dest_folder + "vggsound_train.csv")
    writecsv(valid, dest_folder + "vggsound_dev.csv")
    writecsv(test, dest_folder + "vggsound_test.csv")

