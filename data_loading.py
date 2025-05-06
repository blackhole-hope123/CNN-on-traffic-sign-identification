from utils import load_train_and_test_data
import numpy as np

IMG_WIDTH=30
IMG_HEIGHT=30

x_train, y_train, x_test, y_test = load_train_and_test_data("data", IMG_WIDTH, IMG_HEIGHT)
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
np.savez("traffic_data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_train_mean=x_train_mean, x_train_std=x_train_std)