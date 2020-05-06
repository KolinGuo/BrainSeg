import tensorflow as tf
import keras

# upsample function utilizing tf.keras.layers.Conv2D, size=4, strides=2, default set to be 2x resolution
# upsample function, the size is determined by factor of images, strides is 2 * factor - factor % 2
def upsample(filters, size=4, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def fcn_model(classes=3, drop_out_rate=0.2, bn=True):
    # use dropout and bacth normalization to prevent overfitting and help to do quick convergence
    # activation layer is added to incorporate non-linearity
    
    # input layer has variable length in width and height, tested on both 512, 1024
    input_imgs = tf.keras.layers.Input(shape=(None, None, 3))
    
    # All the kernel size, filter, stride are based on comparson paper, maxpooling layer is based on original FCN paper
    
    # First conv layer + max pooling
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same')(input_imgs)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    x = tf.keras.layers.Activation('relu')(x)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Second conv layer + max pooling
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(pool1)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Third conv layer + max pooling
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(pool2)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Forth conv layer + max pooling
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(pool3)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)  
    
    # Fifth conv layer + max pooling
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=11, strides=1, padding="same")(pool4)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    pool5 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # build the fully connected layer using 1*1 convolutional layer
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding="same")(pool5)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    conv6 = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(conv6)
    x = tf.keras.layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=bn)
    conv7 = tf.keras.layers.Activation('sigmoid')(x)

    # upsampling conv7 to 4x times and upsample pool4 to 2x times
    up_conv7 = upsample(filters=classes, size=8, strides=4)(conv7)
    up_pool4 = upsample(filters=64)(pool4)
    
    # Concatenate two resolutions
    fuse_1 = tf.keras.layers.Concatenate()([up_conv7, up_pool4])
    fuse_2 = tf.keras.layers.Concatenate()([fuse_1, pool3])
    
    prob = upsample(filters=classes, size=16, strides=8)(fuse_2)
    model = tf.keras.Model(inputs=input_imgs, outputs=prob)
    
    print(model.summary())
    print("FCN model building completes")
    
    return model

if __name__ == '__main__':
    model = fcn_model(classes=3)