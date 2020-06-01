import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print('Train on', str(int(len(labels) * (1 - TEST_SIZE))), 'samples')

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # raise NotImplementedError
    images = []
    labels = []
    # get string to the current working directory
    cwd = str(os.getcwd())
    # get the current directory dynamically with os.path.join
    dir_str = os.path.join(cwd, data_dir)
    for file in range(NUM_CATEGORIES):
        file_dir = os.path.join(dir_str, str(file))
        for image_file in os.listdir(file_dir):
            # get the path to the image
            file_path = os.path.join(file_dir, image_file)
            # using cv2, read and resize the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            np_image_data = np.asarray(img)
            # add the resized image and label to our images and labels arrays
            images.append(np_image_data)
            labels.append(file)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # raise NotImplementedError
    model = tf.keras.models.Sequential([

        # First Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.BatchNormalization(
            axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.BatchNormalization(
            axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.BatchNormalization(
            axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all NUM_CATEGORIES
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()