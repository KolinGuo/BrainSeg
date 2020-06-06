"""FCN Model"""
# pylint: disable=invalid-name

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def fcn_model(classes=3, drop_out_rate=0.2, bn=True) -> keras.Model:
    """Use dropout and bacth normalization to prevent overfitting
        and help to do quick convergence
       activation layer is added to incorporate non-linearity
    """

    def upsample(filters, size=4, strides=2, apply_dropout=False):
        """Upsample function utilizing layers.Conv2D, size=4, strides=2,
               default set to be 2x resolution
           Upsample function, the size is determined by factor of images,
               strides is 2 * factor - factor % 2
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        result = keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=strides,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))

        result.add(layers.BatchNormalization())

        if apply_dropout:
            result.add(layers.Dropout(0.5))

        result.add(layers.ReLU())

        return result

    # input layer has variable length in width and height, tested on both 512, 1024
    input_imgs = layers.Input(shape=(None, None, 3))

    # All the kernel size, filter, stride are based on comparson paper
    #   maxpooling layer is based on original FCN paper

    # First conv layer + max pooling
    x = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same')(input_imgs)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    x = layers.Activation('relu')(x)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Second conv layer + max pooling
    x = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(pool1)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    x = layers.Activation('relu')(x)

    pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Third conv layer + max pooling
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(pool2)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    x = layers.Activation('relu')(x)

    pool3 = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Forth conv layer + max pooling
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(pool3)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    x = layers.Activation('relu')(x)

    pool4 = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Fifth conv layer + max pooling
    x = layers.Conv2D(filters=1024, kernel_size=11, strides=1, padding="same")(pool4)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    x = layers.Activation('relu')(x)

    pool5 = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # build the fully connected layer using 1*1 convolutional layer
    x = layers.Conv2D(filters=512, kernel_size=1, strides=1, padding="same")(pool5)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    conv6 = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(conv6)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.BatchNormalization()(x, training=bn)
    conv7 = layers.Activation('sigmoid')(x)

    # upsampling conv7 to 4x times and upsample pool4 to 2x times
    up_conv7 = upsample(filters=classes, size=8, strides=4)(conv7)
    up_pool4 = upsample(filters=64)(pool4)

    # Concatenate two resolutions
    fuse_1 = layers.Concatenate()([up_conv7, up_pool4])
    fuse_2 = layers.Concatenate()([fuse_1, pool3])

    # add three layers of convolution to perform additional learning on skip archiecture
    conv8 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(fuse_2)
    conv9 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(conv8)
    conv10 = layers.Conv2D(filters=16, kernel_size=1, strides=1, padding="same")(conv9)

    prob = upsample(filters=classes, size=16, strides=8)(conv10)
    model = keras.Model(inputs=input_imgs, outputs=prob)

    print(model.summary())
    print("FCN model building completes")

    return model

if __name__ == '__main__':
    fcn_model = fcn_model(classes=3)
    keras.utils.plot_model(fcn_model, 'FCN.png', show_shapes=True)
# pylint: enable=invalid-name
