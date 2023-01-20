import tensorflow as tf
import keras.backend as K
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred, smooth=1):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  return 1- K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def weighted_cross_entropy_fn(y_true, y_pred, weights = [1.0, 200.0]):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  weights_v = tf.where(tf.equal(y_true, 1), weights[1], weights[0])
  ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
  loss = K.mean(tf.multiply(ce, weights_v))
  return loss
    

def ComboLoss(targets, inputs, smooth=1e-6):    
    return (0.3 * dice_loss(targets, inputs)) + (0.7 * weighted_cross_entropy_fn(targets, inputs))