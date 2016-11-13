from tqdm import tqdm
import math
import tensorflow as tf
import numpy as np

def build_conv(input_matrix, chann, n_filter, win, strides, pool):
    num_filters = n_filter
    winx = win
    winy = win
    W = tf.Variable(tf.truncated_normal([winx, winy, chann, num_filters],
                                         stddev = 1./math.sqrt(winx * winy)))
    b = tf.Variable(tf.constant(0.1, shape = [num_filters]))
    xw = tf.nn.conv2d(input_matrix, W,
                       strides = [1, strides, strides, 1], # image, x, y, channel
                       padding = 'SAME')
    h = tf.nn.relu(xw + b)

    # pooling
    p = tf.nn.max_pool(h,
                       ksize =   [1, pool, pool, 1],
                       strides = [1, pool, pool, 1],
                       padding = "VALID")

    keep_prob = tf.placeholder("float")
    drop = tf.nn.dropout(h, keep_prob)

    return p

data = np.load('human_with_label.npz')
train = data['imgs']/255.
labels = data['labels']

indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.2)
test_idx, training_idx = indices[:valid_cnt], \
                         indices[valid_cnt:]

test, train = train[test_idx, :], \
              train[training_idx, :]
label_test, label_train = labels[test_idx, :], \
                          labels[training_idx, :]

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 100 * 100])
y_ = tf.placeholder("float", [None, 2])

# Reshape input image
# undefined, width px, height px, 1 channel (grayscale)
x_im = tf.reshape(x, [-1, 100, 100, 1])

# Conv layer 1
p1 = build_conv(x_im, 1, 10, 15, 3, 2)

# Conv layer 2
p2 = build_conv(p1, 10, 15, 6, 2, 2)

# reshape to use dense layer
ps = np.product([ s.value for s in p2.get_shape()[1:] ])
pf = tf.reshape(p2, [-1, ps])

##############
# Dense layers

# Hidden layer 1.
num_hidden_1 = 60
W3 = tf.Variable(tf.truncated_normal([ps, num_hidden_1],
                                     stddev = 1./math.sqrt(ps)))
b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden_1]))
h3 = tf.nn.relu(tf.matmul(pf, W3) + b3)

# Output layer.
W4 = tf.Variable(tf.truncated_normal([num_hidden_1, 2],
                                     stddev = 1./math.sqrt(2)))
b4 = tf.Variable(tf.constant(0.1, shape=[2]))

# End of defining model.

####################################
# Start session after building model.
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(h3, W4) + b4)

# Cost function.
learning_rate = tf.placeholder(tf.float32, shape=[])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        y + 1e-50, y_))
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(y, 1),
                              tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(
    correct_prediction, "float"))

epochs = 10000
train_acc = np.zeros(epochs//10)
test_acc = np.zeros(epochs//10)

for i in tqdm(range(epochs)):
    if i % 10 == 0:
        A = accuracy.eval(feed_dict={
            x: train.reshape([-1, 100 * 100]),
            y_: label_train})
        train_acc[i//10] = A
        A = accuracy.eval(feed_dict={
            x: test.reshape([-1, 100 * 100]),
            y_: label_test})
        test_acc[i//10] = A
    train_step.run(feed_dict = {
        x: train.reshape([-1, 100 * 100]),
        y_: label_train,
        learning_rate: 0.1
    })
    train_step.run(feed_dict = {
        x: train.reshape([-1, 100 * 100]),
        y_: label_train,
        learning_rate: 0.01
    })
    train_step.run(feed_dict = {
        x: train.reshape([-1, 100 * 100]),
        y_: label_train,
        learning_rate: 0.001
    })

print(train_acc[-1])
print(test_acc[-1])

