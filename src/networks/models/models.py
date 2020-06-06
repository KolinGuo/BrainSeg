"""Model Script Used By train.py and predict.py"""
from tensorflow import keras

from .FCN import fcn_model
from .UNet import unet_model_no_pad, unet_model_zero_pad

def get_model(model_name: str) -> keras.Model:
    """Get the keras.Model based on input"""
    if model_name == 'UNet_No_Pad':
        return unet_model_no_pad(output_channels=3)
    if model_name == 'UNet_No_Pad_3Layer':
        return unet_model_no_pad(output_channels=3,
                                 unet_layers=3,
                                 model_name='UNet_No_Pad_3Layer')
    if model_name == 'UNet_Zero_Pad':
        return unet_model_zero_pad(output_channels=3)
    if model_name == 'UNet_Zero_Pad_3Layer':
        return unet_model_zero_pad(output_channels=3,
                                   unet_layers=3,
                                   model_name='UNet_Zero_Pad_3Layer')
    if model_name == 'FCN':
        return fcn_model(classes=3, bn=True)
    raise ValueError('Unknown model')
