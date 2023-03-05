import tensorflow as tf
import ssl
import numpy as np
from keras.utils.vis_utils import plot_model
ssl._create_default_https_context = ssl._create_unverified_context

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)#expand the data to the 4 dimention



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.build(x_train.shape)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()


