# coding:utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

train_dir = 'E:\\Program\\python\\zhizibei\\speech-spectrograms\\train'
test_dir = 'E:\\Program\\python\\zhizibei\\speech-spectrograms\\test'
label_dir = 'E:\\Program\\python\\zhizibei\\speech-spectrograms\\train_labels.csv'


class exam:
    def __init__(self, train_dir, test_dir, label_dir):
        self.shape = (128, 173)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.label_dir = label_dir

    def label_reads(self):
        label = pd.DataFrame(pd.read_csv(self.label_dir))
        label = list(label['accent'])
        return label

    def img_input(self, img_dir):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            horizontal_flip=True
        )
        train_generator = train_datagen.flow_from_directory(
            directory=img_dir,
            target_size=(128, 173),
            batch_size=32,
            color_mode='grayscale',
            classes=self.label_reads(),
        )
        return train_generator

    def create_model(self, ):
        model = Sequential()
        model.add(Dense(10000, activation='relu', input_shape=(22144,)))
        model.add(Dropout(0.2))
        model.add(Dense(5000, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        return model

    def train_model_and_predict(self, x_train, y_train, batch_size, epochs, x_test, y_test):
        model = self.create_model().fit(x_train, y_train,
                                        batch_size=batch_size,
                                        verbose=1,
                                        validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=1)
        print('Total loss on test set: ', score[0])
        print('Accuracy of testing set: ', score[1])

        result = model.predict_classes(x_test)
        print(result)


test_1 = exam(train_dir, test_dir, label_dir)
# print(test_1.label_reads())
result = test_1.create_model().fit_generator(
    test_1.img_input(train_dir),
    steps_per_epoch=1000,
    epochs=10,
)

