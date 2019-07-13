import cv2
import os

path = "./datasets/uniqlo"
img_list = os.listdir(path)
if ".DS_Store" in img_list:
    img_list.remove(".DS_Store")

out_dir = "uniqlo_test"
if not (os.path.exists(os.path.join("./datasets", out_dir))):
    os.mkdir(os.path.join("./datasets", out_dir))

for img in img_list:
    tgt_img = cv2.imread(os.path.join(path, img))

    # convert an image into gray scale
    gray_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

    # get a binary inverse mask
    _, mask_inverse = cv2.threshold(gray_img, 150,255, cv2.THRESH_BINARY)

    # get a binary mask
    mask = cv2.bitwise_not(mask_inverse)

    # convert a mask into 3 channels
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # apply bitwise and on mask
    masked_img = cv2.bitwise_and(tgt_img, mask_rgb)

    # replace the cut_out parts with white
    mskd_img_replace_white = cv2.addWeighted(masked_img, 1, cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB), 1, 0)

    cv2.imwrite(os.path.join(os.path.join("./datasets", out_dir), "cpied"+img), mskd_img_replace_white)
