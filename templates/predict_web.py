from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

env = Environment(loader=FileSystemLoader("./", encoding="utf8"))
tpl = env.get_template("index.html")


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        f = request.files["file"]  # アップロードされた画像を保存
        file_path = os.path.join("../datasets/test", datetime.now().strftime("%Y/%m/%d/%H/M%/%S") + ".jpg")
        f.save(file_path)

        result_dir = '../model_detail'
        img_height, img_width = 150, 150
        channels = 3

        # VGG16
        input_tensor = Input(shape=(img_height, img_width, channels))
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # VGG16とFCを接続
        model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

        # 学習済みの重みをロード
        model.load_weights(os.path.join(result_dir, 'vgg16_fine.h5'))

        model.compile(loss='binary_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        # 画像を読み込んで4次元テンソルへ変換
        img = image.load_img(file_path, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = x / 255.0

        # クラスを予測
        pred = model.predict(x)[0]
        if pred > 0.5:
            predict = "UNIQLO"
        else:
            predict = "ヨウジヤマモト"
        return render_template("index.html", filepath=file_path, predict=predict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
