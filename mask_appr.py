import cv2
import os

path = "./datasets/uniqlo"
img = "00000243.jpg"

tgt_img = cv2.imread(os.path.join(path, img))

# convert an image into gray scale
gray_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

# get a binary inverse mask
_, mask_inverse = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)

# get a binary mask
mask = cv2.bitwise_not(mask_inverse)

# convert a mask into 3 channels
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# apply bitwise and on mask
masked_img = cv2.bitwise_and(tgt_img, mask_rgb)

# replace the cut_out parts with white
mskd_img_replace_white = cv2.addWeighted(masked_img, 1, cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB), 1, 0)

cv2.imwrite(os.path.join("datasets/uniqlo_test", "cpied"+img), mskd_img_replace_white)