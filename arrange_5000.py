import os
import re
import random
import shutil

path = "./datasets/yoji_expand"
img_list = os.listdir(path)

out_dir = "yoji_expand_5000"
if not (os.path.exists(os.path.join("./datasets", out_dir))):
    os.mkdir(os.path.join("./datasets", out_dir))

for i in range(5000):
    index = re.search(".jpg", img)
    if index:
        shutil.move()


while count > 5000:
    chosen_img = random.choice(img_list)
    if chosen_img != ".DS_Store":
        os.remove(os.path.join(path, chosen_img))