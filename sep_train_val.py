import os
import random
import shutil

class_name = "yoji"
in_path = f"./datasets/{class_name}_expand"

out_train = f"./datasets/main/train/{class_name}"
out_val = f"./datasets/main/validation/{class_name}"

img_list = os.listdir(in_path)
print(len(img_list))

random.shuffle(img_list)
for i in range(4500):
    if img_list[i] != ".DS_Store":
        shutil.move(os.path.join(in_path, img_list[i]), out_train)

for j in range(4500, 5000):
    if img_list[j] != ".DS_Store":
        shutil.move(os.path.join(in_path, img_list[j]), out_val)