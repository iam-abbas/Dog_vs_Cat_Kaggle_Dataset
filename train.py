import cv2
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from datetime import datetime


img_height = 128
img_width = 128

path = "./train"

file = os.listdir(path)
index = []
images = []

# image size and channels
channels = 3
n_inputs = img_width * img_height * channels

# First convolutional layer
conv1_fmaps = 96  # Number of feature maps created by this layer
conv1_ksize = 4  # kernel size 3x3
conv1_stride = 2
conv1_pad = "SAME"

# Second convolutional layer
conv2_fmaps = 192
conv2_ksize = 4
conv2_stride = 4
conv2_pad = "SAME"

# Third layer is a pooling layer
pool3_fmaps = conv2_fmaps  # Isn't it obvious?

n_fc1 = 192  # Total number of output features
n_outputs = 2

validation_number = 0.12


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def next_batch(num):
    index = []
    images = []
    # Data set Creation
    # print("Creating batch dataset "+str(num+1)+"...")
    for f in (range(num * batch_size, (num+1)*batch_size)):
        if file[f].find("dog"):
            index.append(np.array([0, 1])) # Cat
        else:
            index.append(np.array([1, 0])) # Dog
        image = cv2.imread(path + "/" + file[f])
        image = cv2.resize(image, (img_width, img_height), 0, 0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        images.append(image)

    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = images / 255

    # print("\nBatch "+str(num+1)+" creation finished.")
    return [images, index]

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, img_height, img_width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")


conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")
# conv1 = tf.nn.dropout(conv1, 0.25)
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")
# conv2 = tf.nn.dropout(conv2, 0.25)

n_epochs = 10
batch_size = 250

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 8 * 8])
    # pool3_flat = tf.nn.dropout(pool3_flat, 0.50)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    # fc1 = tf.nn.dropout(fc1, 0.50)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    loss_summary = tf.summary.scalar('log_loss', loss)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

with tf.name_scope("init_and_save"):
    saver = tf.train.Saver()

logdir = log_dir("Dog_Cat_cnn")
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


with tf.Session() as sess:
    init.run()
    for epoch in tqdm(range(n_epochs)):
        for iteration in tqdm(range(24000 // batch_size)):
            X_batch, y_batch = next_batch(iteration)
            y_batch = np.argmax(y_batch, axis=1)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            print("Epoch:", epoch+1, "Train accuracy:", acc_train)
            total_avg = 0
            for i in range(24000//batch_size, 25000//batch_size):
                X_val, y_val = next_batch(i)
                acc_test = accuracy.eval(feed_dict={X: X_val, y: np.argmax(y_val, axis=1)})
                total_avg += acc_test
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_batch, y: y_batch})
            file_writer.add_summary(accuracy_summary_str, iteration*(epoch+1))
            file_writer.add_summary(loss_summary_str, iteration*(epoch+1))

            print("\t Test accuracy:{0:.3f}".format(total_avg*batch_size/1000))
            save_path = saver.save(sess, "DogvsCat_model.ckpt")
    file_writer.close()

