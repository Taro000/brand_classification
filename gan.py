import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
from keras import backend as K


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5,
                            subsample=(2, 2),
                            border_mode='same',
                            input_shape=(3, 150, 150)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(256, 5, 5, subsample=(2, 2), dim_ordering="th"))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height * rows, width * cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width * i:width * (i + 1), height * j:height * (j + 1)] = image[0, :, :]
    return combined_image


BATCH_SIZE = 32
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = './model_detail/gen_img'  # 生成画像の保存先
SAVE_WEIGHT_PATH = "./model_detail"
TRAIN_IMG_PATH = "./datasets/uniqlo_test"


def train():
    img_list = os.listdir(TRAIN_IMG_PATH)
    if ".DS_Store" in img_list:
        img_list.remove(".DS_Store")
    X_train = []
    for img in img_list:
        img = load_img(os.path.join(TRAIN_IMG_PATH, img), target_size=(150, 150))
        ary_img = img_to_array(img)
        X_train.append(ary_img)
    X_train = np.array(X_train)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape(X_train.shape[0], 3, X_train.shape[1], X_train.shape[2])
    else:
        pass
    print(X_train.shape)

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            # 生成画像を出力
            if index % 500 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)) \
                    .save(GENERATED_IMAGE_PATH + "%04d_%04d.png" % (epoch, index))

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

        generator.save_weights(os.path.join(SAVE_WEIGHT_PATH, 'generator.h5'))
        discriminator.save_weights(os.path.join(SAVE_WEIGHT_PATH, 'discriminator.h5'))


if __name__ == "__main__":
    train()
