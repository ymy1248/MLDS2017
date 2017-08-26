import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.WARN)

tf.app.flags.DEFINE_bool("train", False, "if true, train model")
tf.app.flags.DEFINE_bool("pred", False, "if true, predict test")
flags = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

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

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return x, y

# read testing data from test file
def read_test_data(file_name = "data"):
    test_image_path = os.path.join(file_name, "test-image")

    with open(test_image_path, "rb") as f:
        f.seek(16)
        x_test = np.fromfile(f, np.uint8).reshape((10000, 28, 28))

    x_test = np.asarray(x_test, dtype=np.float32)
    return x_test

# show number image
def show_image(image):
    plt.imshow(x[0], cmap="gray_r")
    plt.show()

def cnn_model_fn(features, labels, mode):

    # InputLayer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [3, 3],
            padding = "valid",
            activation = tf.nn.relu)


    # Pooling Layer #1
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 64,
            kernel_size = [3, 3],
            padding = "valid",
            activation = tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    dropout_conv2 = tf.layers.dropout(
        inputs=pool2,
        rate=0.25,
        training=mode == learn.ModeKeys.TRAIN)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])

    dense1 = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
            inputs=dense1,
            rate=0.5,
            training=mode == learn.ModeKeys.TRAIN)

    # dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu)
    # dropout2 = tf.layers.dropout(
    #         inputs=dense2,
    #         rate=0.1, 
    #         training=mode == learn.ModeKeys.TRAIN)

    # dense3 = tf.layers.dense(inputs=dropout2, units=256, activation=tf.nn.relu)
    # dropout3 = tf.layers.dropout(
    #         inputs=dense3,
    #         rate=0.4, 
    #         training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout1, units=10)
    
    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="RMSProp")

    # Generate Predictions
    predictions = tf.argmax(input=logits, axis=1)
    # predictions = {
    #         "classes": tf.argmax(input=logits, axis=1),
    #         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #         }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

# main function
def main(unused_argv):
    if flags.train:
        x, y = read_train_data()
        train_len = int(len(x)*0.9)
        x_t = x[:train_len]
        y_t = y[:train_len]
        x_v = x[train_len:]
        y_v = y[train_len:]
        mnist_classifier = learn.Estimator(
                model_fn=cnn_model_fn,
                model_dir="/tmp/cnn_mnist_model",
                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
        
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log,
                every_n_iter=50)

        validation_matrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        }

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            x=x_v,
            y=y_v,
            every_n_steps=50,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=500)

        mnist_classifier.fit(
                x=x_t,
                y=y_t,
                batch_size=256,
                steps=10000,
                monitors=[validation_monitor])

        accuracy_score = mnist_classifier.evaluate(
                x=x_v,
                y=y_v,)
        print(accuracy_score)

    if flags.pred:
        import csv
        x_test = read_test_data()

        mnist_classifier = learn.Estimator(
                model_fn=cnn_model_fn,
                model_dir="/tmp/cnn_mnist_model",)

        ans = np.array(list(mnist_classifier.predict(
                x=x_test,
                batch_size=64,
                as_iterable=True,)))
        with open("ans.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["id", "label"])
            for i in range(len(ans)):
                writer.writerow([i, ans[i]])

        
        # print("accuracy of mnist_classifier: {0:f}".format(accuracy_score))

        #metrics = {
        #        "accuracy": learn.MetricSpec(
        #            metric_fn=tf.metrics.accuracy,
        #            prediction_key="classes"),
        #        }

if __name__ == "__main__":
    tf.app.run()
