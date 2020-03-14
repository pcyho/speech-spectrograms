# coding:utf-8
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob
import time


train_dir = './train'
test_dir = './test'
label_dir = './train_labels.csv'
img_shape = (128, 173)


def input_loads():
    df = pd.read_csv(label_dir)
    df['file_id'] = df['file_id'].astype('str')
    df['file_id'] = df['file_id'].apply(lambda x: x + '.png')

    df['accent'] = df['accent'].astype('str')
    # df['accent'] = df['accent'].map(onehot)

    # label = pd.DataFrame(df['accent'].map(onehot).tolist(), columns=['0', '1', '2'])
    # df = pd.concat([df['file_id'], label], axis=1)
    print(df.head())

    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_dir,
        x_col='file_id',
        y_col='accent',
        class_mode='categorical',
        target_size=(128, 174),
        batch_size=32
    )
    return train_generator


def create_mode():
    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(128, 174, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=RMSprop(1e-4),
                  metrics=['accuracy'])
    return model


def predict_img():
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=(128, 174),
        batch_size=10,
    )
    return test_generator


model = create_mode()

model.fit_generator(generator=input_loads(),
                    steps_per_epoch=100,
                    epochs=11,
                    verbose=1,
                    )

f_names = ['./test/%d.png' % i for i in range(20000, 25377)]

img = []
result = []
for i in range(len(f_names)):
    images = load_img(f_names[i], target_size=(128, 174))
    x = img_to_array(images)
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    result.append(y.tolist()[0])
    print('loading no.%s image' % i)

a = [result[i].index(max(result[i])) for i in range(len(result))]
test_id = pd.DataFrame(pd.read_csv('submission_format.csv'))
a = pd.concat([test_id['file_id'], pd.DataFrame(a, columns=['accent'])['accent']], axis=1)
pd.DataFrame(a).to_csv('result%d.csv')
print('\nfinish`````````````````````````````')


