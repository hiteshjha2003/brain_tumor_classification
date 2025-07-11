# focal_loss.py
import tensorflow as tf
from tensorflow.keras.losses import Loss

class FocalLoss(Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = tf.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=1)
