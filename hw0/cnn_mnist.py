import os
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_bool("pre", False, "if true, preprocess mnist data")
flags = tf.app.flags.FLAGS

def read_mnist(path = ".",
        x_path = "train-images-idx3-ubyte",
        y_path = "train-labels-idx1-ubyte"):
    import struct

    x_path = os.path.join(path, x_path)
    y_path = os.path.join(path, y_path)

    with open(y_path, "rb") as y_file:
        magic, num = struct.unpack(">II", y_file.read(8))
        lbl = np.fromfile(y_file, dtype=np.int8)

    with open(x_path, "rb") as x_file:
        maginc, num, rows, cols = struct.unpack(">IIII", x_file.read(16))
        img = np.fromfile(x_file, dtype=np.uint8).reshape(len(lbl), row, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in xrange(len(lbl)):
        yield get_img(i)

def main(unused_argv):
    if flags.pre == True:
        mnist = read_mnist()

if __name__ == "__main__":
    tf.app.run()
