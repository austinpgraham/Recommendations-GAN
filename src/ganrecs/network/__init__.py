#!/usr/bin/env python3
from ganrecs.network.discriminator import Discriminator

from ganrecs.network.generator import Generator

class GAN():

    def __init__(self, d, d_optimzier, g, g_optimizer):
        self.discriminator_input = d
        self.discriminator_optimizer = d_optimizer
        self.generator_input = g
        self.generator_optimizer = g_optimizer

def gan(dis_arch, gen_arch):
    Z = tf.placeholder(tf.float32, shape=[None, gen_arch[0]])
    X = tf.placeholder(tf.float32, shape=[None, dis_arch[0]])
    g = Generator(Z)
    d = Discriminator(X, g.prob)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_real, labels=tf.ones_like(d.logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=tf.zeroes_like(d.logit_fake)))
    d_total_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=tf.zeroes_like(d.logit_fake)))
    d_opt = tf.train.AdamOptimizer().minimize(d_total_loss, var_list=d.get_var_list())
    g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g.get_var_list())
    return GAN(X, d_opt, Z, g_opt)
