import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import Config as cf


class DownscaleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = tf.keras.layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.us = tf.keras.layers.UpSampling2D((2, 2))
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = tf.keras.layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


class SegmentationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        f = [16, 32, 64, 128, 256]
        self.downscale_blocks = [
                                DownscaleBlock(f[0]),
                                DownscaleBlock(f[1]),
                                DownscaleBlock(f[2]),
                                DownscaleBlock(f[3]),
                                ]
        self.bottle_neck_block = BottleNeckBlock(f[4])
        self.upscale_blocks = [
                               UpscaleBlock(f[3]),
                               UpscaleBlock(f[2]),
                               UpscaleBlock(f[1]),
                               UpscaleBlock(f[0]),
                              ]
        self.conv_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="tanh")
        self.custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def calculate_loss(self, target, pred):
        loss = self.custom_loss(target, pred)
        return loss

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch_data):
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4)


if __name__ == "__main__":
    import numpy as np
    empty_img = np.zeros((1, 128, 512, 3))
    empty_img = tf.image.convert_image_dtype(empty_img, tf.float32)

    model = SegmentationModel()

    out = model(empty_img)
    print(out.shape)
