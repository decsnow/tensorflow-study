import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones([4,3])
c = tf.fill([3,], 9.)
print("a:", a)
print("b:", b)
print("c:", c)
print(b+c)
