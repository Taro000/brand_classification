from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils.vis_utils import plot_model


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for j in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (j, loss[j], acc[j], val_loss[j], val_acc[j]))


if __name__ == "__main__":

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(150, 150, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # vgg16_modelはkeras.engine.training.Model
    # top_modelはSequentialとなっている
    # ModelはSequentialでないためadd()がない
    # そのためFunctional APIで二つのモデルを結合する
    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    model.summary()
    plot_model(model, to_file="./model_detail/model.png")

    # layerを表示
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    model.summary()

    # ここでAdamを使うとうまくいかない
    # Fine-tuningのときは学習率を小さくしたSGDの方がよい？
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        './datasets/main/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        './datasets/main/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # 訓練
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=9000,
        nb_epoch=3,
        validation_data=validation_generator,
        nb_val_samples=1000)

    # 結果を保存
    model.save_weights('./model_detail/vgg16_fine.h5')
    save_history(history, './model_detail/history_vgg16_fine.txt')
