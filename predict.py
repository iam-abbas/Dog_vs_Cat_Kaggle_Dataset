import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm

text = open('Submission.csv', "w")
text.write('id;label\n')

path = "./test"
batch_size = 1
img_width = 128
img_height = 128
file =os.listdir(path)


def submitCSV(id,label):
    text.write(id+";"+label+"\n")


def next_batch(num):
    index = []
    images = []
    image_m = []
    # Data set Creation
    # print("Creating batch dataset "+str(num+1)+"...")
    for f in (range(num * batch_size, (num+1)*batch_size)):
        # if file[f].find("dog"):
        #     index.append(np.array([0, 1]))
        # else:
        #     index.append(np.array([1, 0]))
        image = cv2.imread(path + "/" + file[f])
        image = cv2.resize(image, (img_width, img_height), 0, 0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        images.append(image)

    images = np.array(images, dtype=np.uint8)
    images = (images.astype('float32'))/255
    for i in images:
        image_m.append(i)


    # print("\nBatch "+str(num+1)+" creation finished.")
    return [image]

# saver = tf.train.import_meta_graph('DogsvsCat_model.ckpt')
## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
# saver = tf.train.import_meta_graph('dogs-cats-model.meta')
saver = tf.train.import_meta_graph('DogvsCat_model.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("output/Y_proba:0")

## Let's feed the images to the input placeholders
X = graph.get_tensor_by_name("inputs/X:0")
# y_true = graph.get_tensor_by_name("y_true:0")
# y_test_images = np.zeros((1, 2))

# Creating the feed_dict that is required to be fed to calculate Y_proba

for iteration in tqdm(range(len(file)//batch_size)):
    X_batch = next_batch(iteration)
    # X_batch = X_batch.reshape(1, img_width, img_height, 3)
    feed_dict_testing = {X: X_batch}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    if result[0][0] == 0:
        submitCSV(file[iteration].split('.')[0], "0")
    else:
        submitCSV(file[iteration].split('.')[0], "1")

# result is of this format [probabiliy_of_rose probability_of_sunflower]
# print(file[5].split('.')[0])
print("Done")
text.close()
print(file)
