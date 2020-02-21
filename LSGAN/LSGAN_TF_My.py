# Tensorflow implementation of the paper "Least Squares Generative Adversarial Networks"
# https://arxiv.org/abs/1611.04076
# @Author: ZCB
# @Date:   2019-11-13
# @Last Modified by:   ZCB
# @Last Modified time: 2019-12-04

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',    dest='phase',   default='train', help='train or test')

args = parser.parse_args()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 512
learning_rate = 0.001
epoch = 2503
initializer = tf.truncated_normal_initializer(stddev=0.02)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

''' 
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
'''

def generator(z):
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=7 * 7 * 128, activation_fn=tf.nn.relu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm, \
                                                weights_initializer=initializer, scope="g_fc1")
        fc1 = tf.reshape(fc1, shape=[batch_size, 7, 7, 128])
        conv1 = tf.contrib.layers.conv2d(fc1, num_outputs=4 * 64, kernel_size=5, stride=1, padding="SAME", \
                                         activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                         weights_initializer=initializer, scope="g_conv1")
        conv1 = tf.reshape(conv1, shape=[batch_size, 14, 14, 64])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=4 * 32, kernel_size=5, stride=1, padding="SAME", \
                                         activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                         weights_initializer=initializer, scope="g_conv2")

        conv2 = tf.reshape(conv2, shape=[batch_size, 28, 28, 32])
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=1, kernel_size=5, stride=1, padding="SAME", \
                                         activation_fn=tf.nn.tanh, scope="g_conv3")

        return conv3

def discriminator(tensor, reuse=False):
    with tf.variable_scope("discriminator"):
        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=5, stride=2, padding="SAME", \
                                         reuse=reuse, activation_fn=lrelu, weights_initializer=initializer,
                                         scope="d_conv1")
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=5, stride=2, padding="SAME", \
                                         reuse=reuse, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                         weights_initializer=initializer, scope="d_conv2")
        fc1 = tf.reshape(conv2, shape=[batch_size, 7 * 7 * 64])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512, reuse=reuse, activation_fn=lrelu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm, \
                                                weights_initializer=initializer, scope="d_fc1")
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=tf.nn.sigmoid, \
                                                weights_initializer=initializer, scope="d_fc2")

        return fc2


class LSGAN:

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.z_in = tf.placeholder(tf.float32, shape=[batch_size, 100])


        self.g_out = generator(self.z_in)
        self.d_out_fake = discriminator(self.g_out,     reuse=False)
        self.d_out_real = discriminator(self.x_image,  reuse=True)

        self.disc_loss = tf.reduce_sum(tf.square(self.d_out_real - 1) + tf.square(self.d_out_fake)) / 2
        self.gen_loss = tf.reduce_sum(tf.square(self.d_out_fake - 1)) / 2

        self.gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        self.dis_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        self.d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # Only update the weights for the discriminator network.
        self.d_grads = self.d_optimizer.compute_gradients(self.disc_loss, self.dis_variables)
        # Only update the weights for the generator network.
        self.g_grads = self.g_optimizer.compute_gradients(self.gen_loss, self.gen_variables)

        self.update_D = self.d_optimizer.apply_gradients(self.d_grads)
        self.update_G = self.g_optimizer.apply_gradients(self.g_grads)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver.save(self.sess,os.path.join(ckpt_dir, model_name),global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def train(self):
        saver = tf.train.Saver()
        for i in range(epoch):
            #batch = mnist.train.next_batch(batch_size)
            # z_input = np.random.uniform(0, 1.0, size=[batch_size, 100]).astype(np.float32)
            batchs=np.load('../Data/'+str(i)+'.npy')
            z_input  =  np.load('../Noise_Data/'+str(i)+'.npy')
            _, d_loss = self.sess.run([self.update_D, self.disc_loss], feed_dict={self.x: batchs, self.z_in: z_input})

            for j in range(4):
                _, g_loss = self.sess.run([self.update_G, self.gen_loss], feed_dict={self.z_in: z_input})

            print("i: {} / d_loss: {} / g_loss: {}".format(i, np.sum(d_loss) / batch_size, np.sum(g_loss) / batch_size))

            if i % 500 == 0:
                gen_o = self.sess.run(self.g_out, feed_dict={self.z_in: z_input})
                # result = plt.imshow(gen_o[0][:, :, 0], cmap="gray")
                plt.imsave("../Results_LSGAN/" + "{}.png".format(i), gen_o[0][:, :, 0], cmap="gray")
                self.save(saver, i, "../save_para_LSGAN/", "LSGAN_Model")

    def test(self):
        saver = tf.train.Saver()

        load_model_status_DnCNN, global_step3 = self.load(saver, "../save_para_LSGAN/")

        if load_model_status_DnCNN:
            print("[*] Load weights successfully...")

        # z_input = np.random.uniform(0, 1.0, size=[batch_size, 100, 100]).astype(np.float32)
        z_input = np.load('a.npy')
        _, _, N = z_input.shape
        for idx in range(N):
            gen_o = self.sess.run(self.g_out, feed_dict={self.z_in: z_input[:, :, idx]})
            plt.imsave("../Results_LSGAN/" + "{}.png".format(idx), gen_o[0][:, :, 0], cmap="gray")


if __name__ == '__main__':
    lsgan = LSGAN()
    if args.phase == 'train':
           lsgan.train()
    elif args.phase == 'test':
           lsgan.test()