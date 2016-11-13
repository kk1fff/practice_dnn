import tensorflow as tf
import sys

names = tf.train.string_input_producer([sys.argv[1]])

file_content = tf.read_file([sys.argv[1]])
h, w, channels = tf.image.decode_jpeg(file_content, channels=3)
print("w: {}, h: {}".format(w, h))

tf.
