import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# from utils import data_preprocessing


def data_preprocessing(x_train,x_test):
    # mean subtraction and normalization
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_processed = (x_train - mean) / std
    # we subtract the mean of the training data to prevent the model from accessing information from the test data, ensuring a precise evaluation.
    x_test_processed = (x_test - mean) / std
    print("Data preprocessing completed.")
    return (x_train_processed, x_test_processed)


def main():
    """Main function to run the k-NN baseline model."""

    data = np.load("traffic_data.npz")
    X_train, y_train, X_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    X_train, X_test = data_preprocessing(X_train, X_test)


    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]

    # Reshape the training and testing data from 4D to 2D
    # This multiplies the height, width, and channel dimensions together
    X_train = X_train.reshape(num_train_samples, -1)
    X_test = X_test.reshape(num_test_samples, -1)
    print("\nTraining the k-NN model...")
    
    k = 5
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier
    knn_model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Make Predictions and Evaluate the Model
    print("\nEvaluating the model on the test set...")
    y_pred = knn_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the k-NN model (k={k}): {accuracy:.4f}")
    
    # Show a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Show some example predictions
    print("Displaying some test images and their predicted labels...")
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # Reshape the flattened array back to a 28x28 image for display
        image_to_show = X_test[i].reshape(30, 30, 3)
        plt.imshow(image_to_show, cmap='gray')
        plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()