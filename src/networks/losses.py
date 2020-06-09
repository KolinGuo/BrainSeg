"""Custom losses based on tf.keras.losses"""

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K

def get_loss_func(loss_func: str, class_weight, gamma: float) -> losses.Loss:
    """Returns an instance of the requested loss function"""
    if loss_func == 'SCCE':
        return losses.SparseCategoricalCrossentropy(from_logits=True)
    if loss_func == 'BSCCE':
        return BalancedSCCE(class_weight,
                            from_logits=True)
    if loss_func == 'Sparse_Focal':
        return SparseFocalLoss(gamma=gamma,
                               from_logits=True)
    if loss_func == 'Balanced_Sparse_Focal':
        return BalancedSparseFocalLoss(class_weight,
                                       gamma=gamma,
                                       from_logits=True)
    raise ValueError('Unknown loss_func')

class BalancedSCCE(losses.SparseCategoricalCrossentropy):
    """Balanced Sparse Categorical Cross Entropy Loss
    Weighted by class frequency"""
    def __init__(self,
                 class_weight,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='balanced_sparse_categorical_crossentropy'):
        super(BalancedSCCE, self).__init__(from_logits=from_logits,
                                           reduction=reduction,
                                           name=name)
        self.class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Add class weights
        # pylint: disable=no-value-for-parameter
        sample_weight = tf.gather(self.class_weight, y_true)
        # pylint: enable=no-value-for-parameter
        return super(BalancedSCCE, self).__call__(y_true, y_pred, sample_weight)

    def get_config(self):
        config = {'class_weight': self.class_weight}
        base_config = super(BalancedSCCE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def focal_loss(y_true, y_pred, gamma=2.0, from_logits=False, axis=-1):
    """Computes and returns a focal loss"""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=axis)
    else:
        y_pred /= tf.math.reduce_sum(y_pred, axis=axis, keepdims=True)

    epsilon = tf.constant(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    cce = -1 * y_true * tf.math.log(y_pred)

    loss = cce * tf.math.pow(1 - y_pred, gamma)

    return tf.math.reduce_sum(loss, axis=axis, keepdims=False)

def sparse_focal_loss(y_true, y_pred, gamma=2.0, from_logits=False, axis=-1):
    """Computes and returns a sparse focal loss"""
    y_pred = tf.convert_to_tensor(y_pred)
    # pylint: disable=no-value-for-parameter
    y_true = tf.one_hot(y_true, y_pred.shape[axis], axis=axis)
    # pylint: enable=no-value-for-parameter
    return focal_loss(y_true, y_pred,
                      gamma=gamma, from_logits=from_logits, axis=axis)

class SparseFocalLoss(tf.python.keras.losses.LossFunctionWrapper):
    """Sparse Focal Loss
    SCCE Dynamically Weighted by class accuracy"""
    def __init__(self,
                 gamma=2.0,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='sparse_focal_loss'):
        super(SparseFocalLoss, self).__init__(sparse_focal_loss,
                                              reduction=reduction,
                                              name=name,
                                              gamma=gamma,
                                              from_logits=from_logits)

class BalancedSparseFocalLoss(SparseFocalLoss):
    """Balanced Sparse Focal Loss
    Weighted by class frequency"""
    # pylint: disable=too-many-arguments
    def __init__(self,
                 class_weight,
                 gamma=2.0,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='balanced_sparse_focal_loss'):
        super(BalancedSparseFocalLoss, self).__init__(gamma=gamma,
                                                      from_logits=from_logits,
                                                      reduction=reduction,
                                                      name=name)
        self.class_weight = class_weight
    # pylint: enable=too-many-arguments

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Add class weights
        # pylint: disable=no-value-for-parameter
        sample_weight = tf.gather(self.class_weight, y_true)
        # pylint: enable=no-value-for-parameter
        return super(BalancedSparseFocalLoss, self).\
                __call__(y_true, y_pred, sample_weight)

    def get_config(self):
        config = {'class_weight': self.class_weight}
        base_config = super(BalancedSparseFocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
