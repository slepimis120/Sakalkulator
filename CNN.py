import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.regularizers import l2


def prepare_data(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    images = []
    labels = []
    # because training uses numeric labels
    class_mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "Puta": 10,
                     "PodeljenoSa": 11, "Minus": 12, "Plus": 13}

    for image in image_files:
        image_path = os.path.join(path, image)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        label = image.split('_')[0]
        labels.append(class_mapping[label])

    return images, labels, class_mapping


def split_dataset(images, labels, class_mapping):
    x = np.array(images)
    y = to_categorical(labels, num_classes=len(class_mapping))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val


def create_model():
    tf.random.set_seed(42)

    model = models.Sequential()

    # convolutional layers
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.003), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # multi-dim to one-dim (before dense layers)
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.6))

    # model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(14, activation='softmax'))

    return model


def train_model(path):
    images, labels, class_mapping = prepare_data(path)
    x_train, x_val, y_train, y_val = split_dataset(images, labels, class_mapping)
    model = create_model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ImageDataGenerator for data augmentation and preprocessing
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=20,
                                 # width_shift_range=0.2,
                                 # height_shift_range=0.2,
                                 shear_range=0.2,
                                 # zoom_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest')

    datagen_val = ImageDataGenerator(rescale=1. / 255)  # No augmentation for validation data

    train_generator = datagen.flow(x_train, y_train, batch_size=16)
    val_generator = datagen_val.flow(x_val, y_val, batch_size=16)
    # ...

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # train the model
    # history = model.fit(
    #     train_generator,
    #     epochs=200,
    #     batch_size=32,
    #     steps_per_epoch=len(train_generator),
    #     validation_data=val_generator,
    #     validation_steps=len(val_generator),
    #     # callbacks=[early_stopping],
    #     verbose=1
    # )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=400,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1
    )

    return model


def get_model(path):
    model = train_model(path)
    model.save('cnn_model.h5')
    return model


if __name__ == '__main__':
    path = './data/video/training_data'
    model = get_model(path)
    # prepare_data(path)