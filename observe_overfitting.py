import tensorflow as tf
import numpy as np
from utils import get_model, draw_heatmap, load_train_and_test_data, data_preprocessing

EPOCHS = 10
IMG_WIDTH=30
IMG_HEIGHT=30  
  
def observe_overfitting():
    # load the train and test data
    data = np.load("traffic_data.npz")
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    x_train, x_test = data_preprocessing(x_train, x_test)

    # one hot encoding
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # a possible range of dropout_rates and regularizer_strengths for overfitting
    dropout_rates=[0.0, 0.2, 0.4, 0.6]
    regularizer_strengths=[0, 1e-4, 1e-3, 1e-2]
    overfitting=[]

    for dropout_rate in dropout_rates:
        for regularizer_strength in regularizer_strengths:        
            model = get_model(dropout_rate,regularizer_strength, batch_normalization=True, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

            history=model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, shuffle=False)

            # Evaluate neural network performance
            loss,test_accuracy=model.evaluate(x_test, y_test, verbose=2, batch_size=32)
            last_training_accuracy = history.history['accuracy'][-1]
            overfitting.append([dropout_rate,regularizer_strength,test_accuracy-last_training_accuracy])
    draw_heatmap(overfitting, True)
    

if __name__ == "__main__":
    observe_overfitting()