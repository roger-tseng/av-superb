import csv
import os
import yaml
import subprocess

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)
preprocessed_base_path = config["dataset"]["base_path"] + "/UCF-101-VIDEO"

with open('train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train = row

with open('test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test = row

your_train = os.listdir(f"{preprocessed_base_path}/train")
your_test = os.listdir(f"{preprocessed_base_path}/test")

print("Train     :", len(train), "videos")
print("Your Train:", len(your_train), "videos")
print("Test      :", len(test), "videos")
print("Your Test :", len(your_test), "videos")

train_more = list(set(your_train) - set(train))
train_less = list(set(train) - set(your_train))
test_more = list(set(your_test) - set(test))
test_less = list(set(test) - set(your_test))

print()
print("Train More:", train_more)
print("Train Less:", train_less)
print("Test More :", test_more)
print("Test Less :", test_less)

if config["verify"]["delete_more"]:
    for train in train_more:
        subprocess.run(["rm", f"{preprocessed_base_path}/train/{train}"])
    for test in test_more:
        subprocess.run(["rm", f"{preprocessed_base_path}/test/{test}"])
