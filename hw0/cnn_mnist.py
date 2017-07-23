import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn

tf.app.flags.DEFINE_bool("train", False, "if true, train model")
flags = tf.app.flags.FLAGS

# read training data from mnist dataset
def read_train_data(file_name = "data"):
    train_image_path = os.path.join(file_name, "train-images-idx3-ubyte")
    train_label_path = os.path.join(file_name, "train-labels-idx1-ubyte")
    with open(train_image_path, "rb") as f:
        f.seek(16)
        x = np.fromfile(f, np.uint8).reshape((60000, 28, 28))

    with open(train_label_path, "rb") as f:
        f.seek(8)
        y = np.fromfile(f, np.uint8)

    return x, y

# read testing data from test file
def read_test_data(file_name = "data"):
    test_image_path = os.path.join(file_name, "test-image")

    with open(test_image_path, "rb") as f:
        f.seek(16)
        x_val = np.fromfile(f, np.uint8).reshape((10000, 28, 28))

    return x_val


def main(unused_argv):
    if flags.train:
        x, y = read_train_data()
        plt.imshow(x[0], cmap="gray_r")
        plt.show()

if __name__ == "__main__":
    tf.app.run()
