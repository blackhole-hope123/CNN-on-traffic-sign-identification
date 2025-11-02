'''
This module provides utility functions for loading data, preprocessing data,
and providing the Convolutional Neural Network Model.
'''

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt

num_of_categories=43
IMG_WIDTH=30
IMG_HEIGHT=30

def load_data(data_dir, csv_file_path, img_width, img_height):
    images,labels=[],[]
    df=pd.read_csv(csv_file_path)
    df=df[["ClassId","Path"]]
    for row in df.itertuples():
        image_path=os.path.join(data_dir, row.Path)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (img_width, img_height))
        images.append(resized_image)
        labels.append(int(row.ClassId))
        '''print(images)
        print(type(images[0]))
        print(images[0].shape)
        print(labels)'''
    print("Data loaded from ", csv_file_path)
    return (images,labels)

def save_npz(filename, X, y):
    np.savez_compressed(filename, X=X, y=y)


def get_model(regularizer_strength,dropout_rate, batch_normalization, img_width, img_height):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength), input_shape=(img_width, img_height, 3)),
        *([tf.keras.layers.BatchNormalization()] if batch_normalization else []),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        *([tf.keras.layers.BatchNormalization()] if batch_normalization else []),
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        *([tf.keras.layers.BatchNormalization()] if batch_normalization else []),
        tf.keras.layers.Dense(num_of_categories, activation="softmax")
    ])
    model.compile(
        optimizer="nadam",  
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model constructed.")
    return model


# draw the heatmap for overfitting or underfitting
def draw_heatmap(datapoints,title):

    datapoints = np.array(datapoints)

    x_vals = np.unique(datapoints[:, 0])
    y_vals = np.unique(datapoints[:, 1])
    z_vals = datapoints[:, 2].reshape(len(y_vals), len(x_vals))

    plt.imshow(z_vals, 
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            origin='lower', 
            aspect='auto',
            cmap='viridis') 

    plt.colorbar(label='accu(drop,reg)')
    plt.xlabel('dropout_rate')
    plt.ylabel('regularizer_strength')
    plt.title(title)
    plt.show()
