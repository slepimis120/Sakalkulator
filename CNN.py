import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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

    # Define the model
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # model.add(layers.Dense(512, activation='relu', kernel_regularizer='l2'))  # to reduce overfitting
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
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    # train_generator = datagen.flow_from_directory(
    #     path,
    #     target_size=(224, 224),  # Resize images to a consistent size
    #     batch_size=32,
    #     class_mode='categorical'  # Assumes you have one folder per class in your dataset
    # )
    train_generator = datagen.flow(x_train, y_train, batch_size=32)
    # ...

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # train the model
    # history = model.fit(
    #     train_generator,
    #     epochs=1000,
    #     # batch_size=32,
    #     steps_per_epoch=len(train_generator),
    #     validation_data=(x_val, y_val),
    #     # callbacks=[early_stopping],
    #     verbose=1
    # )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=1000,  # Adjust as needed
        batch_size=32,
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