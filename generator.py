import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import decoder
from model import encoder



tf.reset_default_graph()

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2


sampled, mn, sd, x = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)


sess = tf.Session()

sess.run(tf.global_variables_initializer())


_BATCH_SIZE = 128

saver = tf.train.Saver()


saver.restore(sess, save_path="./model")


randoms = [np.random.normal(0, 1, n_latent) for _ in range(100)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

num_row = 10
num_col = 10
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(100):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(imgs[i], cmap='gray')
    
plt.savefig('generated_images.png')
plt.show()