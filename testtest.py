import os

path = "./datasets/yoji_copy"
img_list = os.listdir(path)
if ".DS_Store" in img_list:
    img_list.remove(".DS_Store")
for img in img_list:
    print(img)
print(len(img_list))

X_train = np.transpose(X_train, (0, 3, 1, 2))