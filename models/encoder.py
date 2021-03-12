import tensorflow as tf


class BasicEncoder64x64(tf.keras.Model):
    def __init__(self, encoding_dim=512, dtype=tf.float32, seed=None):
        super(BasicEncoder64x64, self).__init__(dtype=dtype)
        self.encoding_dim = encoding_dim
        self.seed = seed

    def build(self, input_shape):
        init = tf.keras.initializers.glorot_uniform(seed=self.seed)
        self.bottom = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.3),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.3),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.3),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.3)
        ])
        self.top = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.encoding_dim, kernel_initializer=init)
        ])

    def call(self, inputs, training=None):
        out_conv = self.bottom(inputs)
        out = self.top(out_conv)
        return out, out_conv

