import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from typing import Tuple

def unet_model(output_channels: int) -> keras.Model:
    def _Conv_BN_ReLU(inputs: tf.Tensor, filters: int, 
            kernel_size: Tuple[int, int], name: str) -> tf.Tensor:
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='valid', name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)
        outputs = layers.ReLU(name=name+'_relu')(x)
        return outputs

    def _UpConv_BN_ReLU(inputs: tf.Tensor, filters: int, 
            kernel_size: Tuple[int, int], name: str) -> tf.Tensor:
        x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='valid', name=name)(inputs)
        x = layers.BatchNormalization(name=name+'_BN')(x)
        outputs = layers.ReLU(name=name+'_relu')(x)
        return outputs


    x = inputs = keras.Input(shape=(None, None, 3), name='Input')
    #x = inputs = keras.Input(shape=(572, 572, 3), name='Input')    # Testing
    unet_layers = 4
    first_block_filters = 64
    down_stack = []

    # Downsampling steps
    for i in range(unet_layers):
        x = _Conv_BN_ReLU(x, first_block_filters*2**i, (3, 3), f'Conv{2*i+1}')
        x = _Conv_BN_ReLU(x, first_block_filters*2**i, (3, 3), f'Conv{2*i+2}')
        down_stack.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', name=f'MaxPool{i+1}')(x)

    x = _Conv_BN_ReLU(x, first_block_filters*2**unet_layers, (3, 3), f'Conv{2*unet_layers+1}')
    x = _Conv_BN_ReLU(x, first_block_filters*2**unet_layers, (3, 3), f'Conv{2*unet_layers+2}')

    # Upsampling steps
    croppings = [12*2**i - 8 for i in range(unet_layers)]
    down_stack.reverse()
    for i in range(unet_layers):
        x = _UpConv_BN_ReLU(x, first_block_filters*2**(unet_layers-1-i), (2, 2), f'UpConv{i+1}')
        skip = layers.Cropping2D(cropping=croppings[i], name=f'Crop{i+1}')(down_stack[i])
        x = layers.Concatenate(name=f'Concate{i+1}')([skip, x])
        x = _Conv_BN_ReLU(x, first_block_filters*2**(unet_layers-1-i), (3, 3), f'Conv{2*(i+unet_layers+1)+1}')
        x = _Conv_BN_ReLU(x, first_block_filters*2**(unet_layers-1-i), (3, 3), f'Conv{2*(i+unet_layers+1)+2}')


    x = layers.Conv2D(output_channels, (1, 1), strides=1, padding='valid', name='Conv_Final')(x)

    return keras.Model(inputs=inputs, outputs=x, name='UNet')

if __name__ == '__main__':
    model = unet_model(3)
    keras.utils.plot_model(model, 'UNet_model.png', show_shapes=True)

