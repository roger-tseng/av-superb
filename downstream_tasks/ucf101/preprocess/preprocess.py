import os
import subprocess

import ffmpeg
import yaml

train_group = [
    "g08",
    "g09",
    "g10",
    "g11",
    "g12",
    "g13",
    "g14",
    "g15",
    "g16",
    "g17",
    "g18",
    "g19",
    "g20",
    "g21",
    "g22",
    "g23",
    "g24",
    "g25",
]
dev_group = []
test_group = ["g01", "g02", "g03", "g04", "g05", "g06", "g07"]
classIndList = []

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

origin_base_path = config["dataset"]["base_path"] + "/UCF-101"
preprocessed_base_path = config["dataset"]["base_path"] + "/UCF-101-VIDEO"

subprocess.run(["mkdir", f"{preprocessed_base_path}"])
subprocess.run(["mkdir", f"{preprocessed_base_path}/train"])
subprocess.run(["mkdir", f"{preprocessed_base_path}/dev"])
subprocess.run(["mkdir", f"{preprocessed_base_path}/test"])

f = open(f"{origin_base_path}/ucfTrainTestlist/classInd.txt")
for line in f.readlines():
    split = line.split("\n")[0].split(" ")
    classIndList.append([split[0], split[1]])
f.close

progressIdx = 0

for typeName in os.listdir(f"{origin_base_path}"):
    if typeName == "ucfTrainTestlist":
        continue
    for classInd in classIndList:
        if typeName == classInd[1]:
            ind = classInd[0]
            break
    progressIdx += 1
    print(f"{progressIdx}/101: {typeName}")
    for aviFile in os.listdir(f"{origin_base_path}/{typeName}"):
        name = aviFile.split(".")[0]
        group = name.split("_")[2]
        clip = name.split("_")[3]

        input = ffmpeg.input(f"{origin_base_path}/{typeName}/{name}.avi")
        audio = input.audio
        video = input.video
        if group in train_group:
            output = ffmpeg.output(
                audio, video, f"{preprocessed_base_path}/train/{group}_{clip}_{ind}.mp4"
            )
        elif group in dev_group:
            output = ffmpeg.output(
                audio, video, f"{preprocessed_base_path}/dev/{group}_{clip}_{ind}.mp4"
            )
        elif group in test_group:
            output = ffmpeg.output(
                audio, video, f"{preprocessed_base_path}/test/{group}_{clip}_{ind}.mp4"
            )

        try:
            ffmpeg.run(output, quiet=True)
        except:
            pass
