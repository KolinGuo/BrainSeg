import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.python.keras.utils import losses_utils
from typing import List

def get_loss_func(loss_func: str, class_weight) -> losses.Loss:
    if loss_func == 'SCCE':
        return losses.SparseCategoricalCrossentropy(from_logits=True)
    elif loss_func == 'BSCCE':
        return BalancedSCCE(class_weight, from_logits=True)
    else:
        raise ValueError('Unknown loss_func')

class BalancedSCCE(losses.SparseCategoricalCrossentropy):
    """Balanced Sparse Categorical Cross Entropy Loss
    Weighted by class frequency"""
    def __init__(self, 
                 class_weight,
                 from_logits=False, 
                 reduction=losses_utils.ReductionV2.AUTO, 
                 name='balanced_sparse_categorical_crossentropy'):
        super(BalancedSCCE, self).__init__(
                from_logits=from_logits,
                reduction=reduction,
                name=name)
        self.class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.gather(self.class_weight, y_true)
        return super(BalancedSCCE, self).__call__(y_true, y_pred, sample_weight)
