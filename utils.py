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

num_of_categories=43
IMG_WIDTH=30
IMG_HEIGHT=30

# a method suitable for both train data and test data loading
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

# for loading the test and training data
def load_train_and_test_data(data_dir, img_width, img_height):
    train,test="Train.csv","Test.csv"
    files=set([f for f in os.listdir(data_dir)])
    if test in files:
        test_label_path=os.path.join(data_dir, test)
        x_test,y_test=load_data(data_dir, test_label_path, img_width, img_height)
    else:
        raise Exception("Test labels are not available")
    if train in files:
        train_label_path=os.path.join(data_dir, train)
        x_train,y_train=load_data(data_dir, train_label_path, img_width, img_height)
    else:
        raise Exception("Training labels are not available")
    return (x_train,y_train,x_test,y_test)


'''
Another way of importing the training data
'''
'''
def load_training_data(data_dir):
    x_train, y_train = [], []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for i in range(len(folders)):
        label = int(folders[i])
        new_data_dir = os.path.join(data_dir, folders[i])
        files = [f for f in os.listdir(new_data_dir) if os.path.isfile(os.path.join(new_data_dir, f))]
        for j in range(len(files)):
            file_path = os.path.join(new_data_dir, files[j])
            image = cv2.imread(file_path)
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            x_train.append(resized_image)
            y_train.append(label)
    return (x_train, y_train)
'''

def data_preprocessing(x_train,x_test):
    # mean subtraction and normalization
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_processed = (x_train - mean) / std
    # we subtract the mean of the training data to prevent the model from accessing information from the test data, ensuring a precise evaluation.
    x_test_processed = (x_test - mean) / std
    print("Data preprocessing completed.")
    return (x_train_processed, x_test_processed)


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
    print("Model trained.")
    return model


# draw the heatmap for overfitting or underfitting
def draw_heatmap(datapoints,over_or_under):

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
    if over_or_under:
        plt.title('overfitting')
    else:
        plt.title('underfitting')
    plt.show()
