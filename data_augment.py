import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def draw_img(generator, x, out_dir, img_index):
    save_name = "expand_" + str(img_index)

    g = generator.flow(x, batch_size=1, save_to_dir=out_dir, save_prefix=save_name, save_format="jpg")

    for j in range(11):
        g.next()


if __name__ == "__main__":
    in_dir = "uniqlo_test"
    out_dir = "uniqlo_expand"

    if not (os.path.exists(os.path.join("./datasets", out_dir))):
        os.mkdir(os.path.join("./datasets", out_dir))

    images = glob.glob(os.path.join("./datasets", in_dir, "*"))

    generator = ImageDataGenerator(height_shift_range=0.2)

    for i in range(len(images)):
        target_img = load_img(images[i])
        x = img_to_array(target_img)
        x = np.expand_dims(x, axis=0)

        draw_img(generator, x, os.path.join("./datasets", out_dir), i)
