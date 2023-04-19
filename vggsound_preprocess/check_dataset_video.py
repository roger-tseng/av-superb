import argparse
import csv
import random
import os

def getFilename(l):
    filename = "_".join([l[0],str(int(l[1])*1000),str(int(l[1])*1000+10000)+".mp4"])
    return filename

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

    data_folder = args.data
    if data_folder[-1] != "/":  data_folder += "/"

    dest_folder = args.dest
    if dest_folder[-1] != "/":  dest_folder += "/"

    datas = list([])

    existing_data = [list([]) for i in range(num_classes)]

    train = list([])
    valid = list([])
    test = list([])

    with open(args.csv,"r") as csvfile:
        datas = list(csv.reader(csvfile,delimiter=","))

    dictionary = dict([])
    for data in datas:
        dictionary[getFilename(data)] = data

    #existing_files = os.listdir(data_folder)
    
    #for filename in existing_files:
    #    data = dictionary[filename]
    #    if data[-1] == "test":
    #        test.append(data)
    #    else:
    #        train.append(data)

    lsla_audio = [i.split() for i in os.popen('ls {} -la'.format(data_folder.replace("video","audio"))).read().split("\n")[1:-1]]
    exist_audio_files = []
    for file in lsla_audio:
        filename = file[8]
        filesize = int(file[4])
        if filename == '.': continue
        if filename == '..': continue
        if filesize < 50000: continue
        exist_audio_files.append(filename.split('.')[0])


    lsla = [i.split() for i in os.popen('ls {} -la'.format(data_folder)).read().split("\n")[1:-1]]
    print("There're",len(lsla),"files in video directory")

    tooSmallFiles = 0
    noAudioFiles = 0

    for file in lsla:
        filename = file[8]
        filesize = int(file[4])
        if filename == '.': continue
        if filename == '..': continue
        if filesize < 50000:   
            tooSmallFiles += 1
            continue
        if filename.split('.')[0] not in exist_audio_files:
            noAudioFiles += 1
            continue
        """
        data = dictionary[filename]
        if data[-1] == "test":
            test.append(data)
        else:
            train.append(data)
        """
        data = dictionary[filename]
        existing_data[int(data[2])].append(data)


    print("There're {} files too small".format(tooSmallFiles))
    print("There're {} audio files not exist".format(noAudioFiles))

    print("starting split valid and train")


    for i in range(num_classes):
        random.shuffle(existing_data[i])


    for i in range(chosen_class):
        for j in range(len(existing_data[i])):
            if j % 10 == 1:
                valid.append(existing_data[i][j])
            elif j % 10 == 2:
                test.append(existing_data[i][j])
            else:
                train.append(existing_data[i][j])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    print("train/valid/test csv length = {}/{}/{}".format(len(train),len(valid),len(test)))

    """
    valid_index = sorted(random.sample(range(len(train)),len(train)//5),reverse=True)

    for i in valid_index:
        valid.append(train[i])
        del train[i]
    """

    print("writing train/valid/test csv")
    writecsv(train, dest_folder + "vggsound_train.csv")
    writecsv(valid, dest_folder + "vggsound_dev.csv")
    writecsv(test, dest_folder + "vggsound_test.csv")

