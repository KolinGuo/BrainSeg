import numpy as np
import tensorflow as tf
import keras

class focal_loss(keras.losses.Loss):
    # default values set the same as original paper
    def __init__(self, gamma=2.0, alpha=4.0):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
    def call(self, y_true, y_pred):
        epsilon = 1.e-9
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=0)
        return tf.reduce_mean(reduced_fl)

class balanced_focal_loss(keras.losses.Loss):
    # default values set the same as original paper
    def __init__(self, gamma=2.0, alpha=4.0):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
    def call(self, y_true, y_pred):
        epsilon = 1.e-9
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=0)
        total = y_true.flatten().shape[0]
        deno = 0
        for i in range(y_true.shape[-1]):
            deno += 1/np.sum(y_true[:,:,:,i])
        result = []
        for i in range(y_true.shape[-1]):
            result.append(np.array(reduced_fl[:,:,i])*1/np.sum(y_true[:,:,:,i])/deno)
        return tf.reduce_mean(np.array(result))
