"""UNet Model"""
# pylint: disable=invalid-name, too-many-arguments, too-many-locals
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_activation_fn(activation: str, name: str) -> tf.Tensor:
    """Returns the activation function"""
    if activation == 'relu':
        return layers.ReLU(name=name+'_relu')
    if activation == 'elu':
        return layers.ELU(name=name+'_elu')
    raise ValueError('Unknown activation function')

def unet_model_no_pad(output_channels: int,
                      unet_levels=4, first_level_filters=64,
                      kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                      activation='relu', use_dropout=False, dropout_rate=0.2,
                      model_name='UNet_No_Pad') -> keras.Model:
    """Build and return a UNet model with no padding"""
    def _conv_block(inputs: tf.Tensor, filters: int,
                    kernel_size: Tuple[int, int], kernel_initializer: str,
                    activation: str, use_dropout: bool, dropout_rate: float,
                    name: str) -> tf.Tensor:
        """Single conv + batchnorm + activation + (dropout) layer"""
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='valid',
                          kernel_initializer=kernel_initializer, name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)

        outputs = get_activation_fn(activation, name)(x)

        if use_dropout:
            outputs = layers.Dropout(dropout_rate, name=name+'_Drop')(outputs)

        return outputs

    def _upconv_block(inputs: tf.Tensor, filters: int,
                      kernel_size: Tuple[int, int], kernel_initializer: str,
                      activation: str, name: str) -> tf.Tensor:
        """Single upconv + batchnorm + activation layer"""
        x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='valid',
                                   kernel_initializer=kernel_initializer, name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)

        outputs = get_activation_fn(activation, name)(x)

        return outputs


    x = inputs = keras.Input(shape=(None, None, 3), name='Input')
    #x = inputs = keras.Input(shape=(572, 572, 3), name='Input')    # Testing
    down_stack = []

    # Downsampling steps
    for i in range(unet_levels):
        x = _conv_block(x, first_level_filters*2**i, kernel_size, kernel_initializer,
                        activation, use_dropout, dropout_rate, f'Conv{2*i+1}')
        x = _conv_block(x, first_level_filters*2**i, kernel_size, kernel_initializer,
                        activation, False, dropout_rate, f'Conv{2*i+2}')
        down_stack.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=2,
                             padding='valid', name=f'MaxPool{i+1}')(x)

    x = _conv_block(x, first_level_filters*2**unet_levels, kernel_size, kernel_initializer,
                    activation, use_dropout, dropout_rate, f'Conv{2*unet_levels+1}')
    x = _conv_block(x, first_level_filters*2**unet_levels, kernel_size, kernel_initializer,
                    activation, False, dropout_rate, f'Conv{2*unet_levels+2}')

    # Upsampling steps
    croppings = [12*2**i - 8 for i in range(unet_levels)]
    down_stack.reverse()
    for i in range(unet_levels):
        x = _upconv_block(x, first_level_filters*2**(unet_levels-1-i), (2, 2),
                          kernel_initializer, activation, f'UpConv{i+1}')

        skip = layers.Cropping2D(cropping=croppings[i],
                                 name=f'Crop{i+1}')(down_stack[i])
        x = layers.Concatenate(name=f'Concate{i+1}')([skip, x])

        x = _conv_block(x, first_level_filters*2**(unet_levels-1-i),
                        kernel_size, kernel_initializer, activation,
                        use_dropout, dropout_rate, f'Conv{2*(i+unet_levels+1)+1}')
        x = _conv_block(x, first_level_filters*2**(unet_levels-1-i),
                        kernel_size, kernel_initializer, activation,
                        False, dropout_rate, f'Conv{2*(i+unet_levels+1)+2}')

    x = layers.Conv2D(output_channels, (1, 1), strides=1, padding='valid',
                      kernel_initializer=kernel_initializer, name='Conv_Final')(x)

    return keras.Model(inputs=inputs, outputs=x, name=model_name)

def unet_model_zero_pad(output_channels: int,
                        unet_levels=4, first_level_filters=64,
                        kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                        activation='relu', use_dropout=False, dropout_rate=0.2,
                        model_name='UNet_Zero_Pad') -> keras.Model:
    """Build and return a UNet model with zero padding, same-size output"""
    def _conv_block(inputs: tf.Tensor, filters: int,
                    kernel_size: Tuple[int, int], kernel_initializer: str,
                    activation: str, use_dropout: bool, dropout_rate: float,
                    name: str) -> tf.Tensor:
        """Single conv + batchnorm + activation + (dropout) layer"""
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                          kernel_initializer=kernel_initializer, name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)

        outputs = get_activation_fn(activation, name)(x)

        if use_dropout:
            outputs = layers.Dropout(dropout_rate, name=name+'_Drop')(outputs)

        return outputs

    def _upconv_block(inputs: tf.Tensor, filters: int,
                      kernel_size: Tuple[int, int], kernel_initializer: str,
                      activation: str, name: str) -> tf.Tensor:
        """Single upconv + batchnorm + activation layer"""
        x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                   kernel_initializer=kernel_initializer, name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)

        outputs = get_activation_fn(activation, name)(x)

        return outputs


    x = inputs = keras.Input(shape=(None, None, 3), name='Input')
    #x = inputs = keras.Input(shape=(512, 512, 3), name='Input')    # Testing
    down_stack = []

    # Downsampling steps
    for i in range(unet_levels):
        x = _conv_block(x, first_level_filters*2**i, kernel_size, kernel_initializer,
                        activation, use_dropout, dropout_rate, f'Conv{2*i+1}')
        x = _conv_block(x, first_level_filters*2**i, kernel_size, kernel_initializer,
                        activation, False, dropout_rate, f'Conv{2*i+2}')
        down_stack.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=2,
                             padding='same', name=f'MaxPool{i+1}')(x)

    x = _conv_block(x, first_level_filters*2**unet_levels, kernel_size, kernel_initializer,
                    activation, use_dropout, dropout_rate, f'Conv{2*unet_levels+1}')
    x = _conv_block(x, first_level_filters*2**unet_levels, kernel_size, kernel_initializer,
                    activation, False, dropout_rate, f'Conv{2*unet_levels+2}')

    # Upsampling steps
    down_stack.reverse()
    for i in range(unet_levels):
        x = _upconv_block(x, first_level_filters*2**(unet_levels-1-i), (2, 2),
                          kernel_initializer, activation, f'UpConv{i+1}')

        x = layers.Concatenate(name=f'Concate{i+1}')([down_stack[i], x])

        x = _conv_block(x, first_level_filters*2**(unet_levels-1-i),
                        kernel_size, kernel_initializer, activation,
                        use_dropout, dropout_rate, f'Conv{2*(i+unet_levels+1)+1}')
        x = _conv_block(x, first_level_filters*2**(unet_levels-1-i),
                        kernel_size, kernel_initializer, activation,
                        False, dropout_rate, f'Conv{2*(i+unet_levels+1)+2}')

    x = layers.Conv2D(output_channels, (1, 1), strides=1, padding='valid',
                      kernel_initializer=kernel_initializer, name='Conv_Final')(x)

    return keras.Model(inputs=inputs, outputs=x, name=model_name)

if __name__ == '__main__':
    model = unet_model_zero_pad(3)
    keras.utils.plot_model(model, 'UNet_model.png', show_shapes=True)
# pylint: enable=invalid-name, too-many-arguments, too-many-locals
