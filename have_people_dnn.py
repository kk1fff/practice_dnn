from tqdm import tqdm
import math
import tensorflow as tf
import numpy as np

data = np.load('img_with_label.npz')
train = data['imgs']/255.
labels = data['labels']

indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0]*0.1)
test_idx, training_idx = indices[:valid_cnt], \
                         indices[valid_cnt:]

test, train = train[test_idx, :], \
              train[training_idx, :]
label_test, label_train = labels[test_idx, :], \
                          labels[training_idx, :]

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 100*100])
y_ = tf.placeholder("float", [None, 2])

# Hidden layer 1.
num_hidden_1 = 128
W1 = tf.Variable(tf.truncated_normal([100*100, num_hidden_1],
                                     stddev = 1./math.sqrt(100*100)))
b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden_1]))
h1 = tf.sigmoid(tf.matmul(x, W1) + b1)

# Hidden layer 2.
num_hidden_2 = 128
W2 = tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2],
                                     stddev = 1./math.sqrt(num_hidden_1)))
b2 = tf.Variable(tf.constant(0.1, shape=[num_hidden_2]))
h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)

# Output layer.
W3 = tf.Variable(tf.truncated_normal([num_hidden_2, 2],
                                     stddev = 1./math.sqrt(2)))
b3 = tf.Variable(tf.constant(0.1, shape=[2]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

# End of defining model.

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        y + 1e-50, y_))

train_step = tf.train.GradientDescentOptimizer(
    0.02).minimize(cross_entropy)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(y, 1),
                              tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(
    correct_prediction, "float"))

epochs = 1000
train_acc = np.zeros(epochs//10)
test_acc = np.zeros(epochs//10)

for i in tqdm(range(epochs)):
    if i % 10 == 0:
        A = accuracy.eval(feed_dict={
            x: train.reshape([-1, 100*100]),
            y_: label_train})
        train_acc[i//10] = A
        A = accuracy.eval(feed_dict={
            x: test.reshape([-1, 100*100]),
            y_: label_test})
        test_acc[i//10] = A
    train_step.run(feed_dict = {
        x: train.reshape([-1, 100*100]),
        y_: label_train
    })

print(train_acc[-1])
print(test_acc[-1])
