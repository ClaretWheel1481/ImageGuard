import tensorflow as tf
import numpy as np
tf.debugging.set_log_device_placement(True)

# Example operation
a = tf.constant([[1.0, 2.0, 3.0]])
b = tf.reduce_sum(a)
print(b)
