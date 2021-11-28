from model import decoder
from model import encoder
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data




mnist = input_data.read_data_sets('MNIST_data')

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


unreshaped = tf.reshape(dec, [-1, 28*28])

img_loss = Y_flat * tf.log(1e-10 + unreshaped) + (1 - Y_flat) * tf.log(1e-10 + 1 - unreshaped)
img_loss = -tf.reduce_sum(img_loss, 1)

# KL Divergence loss

latent_loss = 1 + sd - tf.square(mn) - tf.exp(sd)
latent_loss = -0.5 * tf.reduce_sum(latent_loss, 1)
    
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

_BATCH_SIZE = 128

saver = tf.train.Saver()


def train(epoch):

    batch_size = int(math.ceil(mnist.train.num_examples / _BATCH_SIZE))


    for s in range(batch_size):

        batch_xs = mnist.train.images[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE].reshape(-1, 28, 28)

        sess.run(optimizer, feed_dict = {X_in: batch_xs, Y: batch_xs, keep_prob: 0.8})
      
    ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch_xs, Y: batch_xs, keep_prob: 1.0})


    print("###########################################################################################################")
    print("Total Loss: {}\n".format(ls))
    print("Reconstruction Loss: {}\n".format(np.mean(i_ls)))
    print("Regularization: {}\n".format(np.mean(d_ls)))


    return np.mean(i_ls), np.mean(d_ls)





def main():

    _EPOCH = 100

    kl_loss = []
    recons_loss = []

    for i in range(_EPOCH):

        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        recons, kl = train(i)
        kl_loss.append(kl)
        recons_loss.append(recons)

    save_path = saver.save(sess, "./model")

    x = range(len(kl_loss))
    y = kl_loss
    z = recons_loss

    plt.plot(x,z, label="Reconstruction Loss")
    plt.plot(x,y, label="KL Divergence")
    plt.legend()
    plt.title('Loss of Reconstruction loss and KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('final_loss.png')



if __name__ == "__main__":
    main()

sess.close()
