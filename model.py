import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math



dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

def encoder(X_in, keep_prob):

    with tf.variable_scope("encoder", reuse=None):
        # Encode input sequence
        cell1 = tf.nn.rnn_cell.LSTMCell(n_latent) 

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, X_in, dtype=tf.float32)
        encoder_output = encoder_output[:, -1, :]

        mn = tf.layers.dense(encoder_output, units=n_latent)
        sd       = 0.5 * tf.layers.dense(encoder_output, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(encoder_output)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
            
    return z, mn, sd, encoder_output



def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=tf.nn.relu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img

