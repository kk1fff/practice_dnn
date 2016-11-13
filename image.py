import tensorflow as tf
import numpy as np
import math

sess = tf.InteractiveSession()
x1 = tf.Variable(tf.truncated_normal([5], mean=3, stddev=1./math.sqrt(5)))
x2 = tf.Variable(tf.truncated_normal([5], mean=-1, stddev=1./math.sqrt(5)))
x3 = tf.Variable(tf.truncated_normal([5], mean=0, stddev=1./math.sqrt(5)))

sess.run(tf.initialize_all_variables())

sqx2 = x2 * x2
print(x2.eval())
print(sqx2.eval())

logx1 = tf.log(x1)
print(x1.eval())
print(logx1.eval())

sigx3 = tf.sigmoid(x3)
print("sigmoid(x3):")
print(x3.eval())
print(sigx3.eval())

w1 = tf.constant(0.1)
w2 = tf.constant(0.2)
sess.run(tf.initialize_all_variables())
n1 = tf.sigmoid(w1*w1 + w2*w2)
print()
print((w1*x1).eval())
print((w2*x2).eval())
print(n1.eval())

