#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from ganrecs.network.discriminator import Discriminator

from ganrecs.network.generator import Generator

class GAN():

    def __init__(self, d, d_optimizer, d_net, d_loss, g, cond, g_optimizer, g_net, g_loss):
        self.discriminator_input = d
        self.discriminator_optimizer = d_optimizer
        self.generator_input = g
        self.generator_optimizer = g_optimizer
        self.discriminator = d_net
        self.generator = g_net
        self.discriminator_loss = d_loss
        self.generator_loss = g_loss
        self.generator_condition = cond

def gan(dis_arch, gen_arch, conditional):
    Z = tf.placeholder(tf.float32, shape=[None, gen_arch[0]], name='noise')
    Y = tf.placeholder(tf.float32, shape=[None, conditional], name='conditional')
    X = tf.placeholder(tf.float32, shape=[None, dis_arch[0]])
    I = tf.concat(values=[Z, Y], axis=1)
    J = tf.concat(values=[X, Y], axis=1)
    gen_arch[0] += conditional
    dis_arch[0] += conditional
    g = Generator(gen_arch, I)
    F = tf.concat(values=[g.prob, Y], axis=1)
    d = Discriminator(dis_arch, J, F)
    print(d.logit_real)
    d_real_labels = tf.ones_like(d.logit_real)
    d_real_labels  = tf.random_uniform([50, int(d.logit_real.shape[1])], minval=0.7, maxval=1.2)
    d_fake_labels = tf.zeros_like(d.logit_fake)
    d_fake_labels = tf.random_uniform([50, int(d.logit_fake.shape[1])], minval=0.0, maxval=0.3)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_real, labels=d_real_labels))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=d_fake_labels))
    d_total_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=d_fake_labels))
    d_opt = tf.train.AdamOptimizer().minimize(d_total_loss, var_list=d.get_var_list())
    g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g.get_var_list())
    return GAN(X, d_opt, d, d_total_loss, Z, Y, g_opt, g, g_loss)
