import tensorflow as tf
import numpy as np

x = tf.placeholder("float")
y = tf.slice(x,[1],[2])

#initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#run
result = sess.run(y, feed_dict={x:[1,2,3,4,5]})
print(result)
