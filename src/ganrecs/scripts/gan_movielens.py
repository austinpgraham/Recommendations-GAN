#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from random import randint

from ganrecs.data import RatingCollection

from ganrecs.network import gan

from surprise import Dataset

import matplotlib.pyplot as plt

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MOVIES_COUNT = 3706
USER_COUNT = 6040
TEST_PERCENT = .2


def sample_Z(m, n):
    return np.random.normal(-1., 1., size=[m, n])


def process_args(args=None):
    parser = argparse.ArgumentParser(description="Test with MNIST data set")
    parser.add_argument('-l', '--location', help='Saved model location')
    parser.add_argument('-n', '--noise', help="Amount of noise to include in network")
    parser.add_argument('-e', '--epochs', help="Number of epochs to run")
    args = parser.parse_args(args)

    assert args.location, "Must provide output location"
    assert args.noise, "Must provide noise"
    assert args.epochs, "Must provide epochs"

    location = os.path.expanduser(args.location)

    if not os.path.exists(location):
        os.makedirs(location)
    else:
        ans = input("Overwrite output directory?: ").upper()
        if ans == 'N' or ans == 'NO':
            print('Exiting...')
            exit()
        shutil.rmtree(location)

    return location, int(args.noise), int(args.epochs)


def get_sample(data, size):
    indices = [randint(1, USER_COUNT) for _ in range(size)]
    used = []
    result = []
    for i in indices:
        while str(i) not in data.keys() or i in used:
            i = randint(1, USER_COUNT)
        used.append(i)
        result.append(list(data[str(i)].values()))
    return np.array(result)


def plot_losses(epochs, d_losses, g_losses):
    xs = [x for x in range(epochs)]
    plt.title('D/G Losses Over Time')
    plt.plot(xs, d_losses, label='Discriminator')
    plt.plot(xs, g_losses, label='Generator')
    plt.legend()
    plt.show()


def main(args=None):
    location, noise, epochs = process_args(args)
    model_path = os.path.join(location, "model.ckpt")
    data = Dataset.load_builtin('ml-100k')
    rc = RatingCollection(data.raw_ratings)

    print("Constructing network...")
    dis_arch = [MOVIES_COUNT, 2000, 1000, 1]
    gen_arch = [noise, 3000, 2000, 3000, MOVIES_COUNT]
    network = gan(dis_arch, gen_arch, MOVIES_COUNT)

    saver = tf.train.Saver()
    d_losses = []
    g_losses = []
    session = tf.Session()
    if os.path.exists(model_path + ".meta"):
        print("Restoring model....")
        saver.restore(session, model_path)
    else:
        session.run(tf.global_variables_initializer())
        print("Starting run...")
        for i in range(len(rc.folds)):
            training_data = {}
            for idx, value in enumerate(rc.folds):
                if idx != i:
                    training_data = {**training_data, **rc._get_matrix(value)}
            for it in range(epochs):
                users = get_sample(training_data, 50)
                _sample = sample_Z(50, noise)
                from pdb import set_trace; set_trace()
                _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss],
                feed_dict={network.discriminator_input: users, network.generator_input: _sample,
                network.generator_condition: users})
                _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss],
                feed_dict={network.generator_input: _sample, network.generator_condition: users})

                if it % 100 == 0:
                    d_losses.append(D_loss_curr)
                    g_losses.append(G_loss_curr)
                    print('Iter: {}'.format(it))
                    print('D loss: {:.4}'.format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    d_losses.append(D_loss_curr)
                    g_losses.append(G_loss_curr)
                    print()

        # print("Saving model to {}".format(location))
        # saver.save(session, model_path)
        # plot_losses(int(epochs / 100), d_losses, g_losses)

        # i = 0
        # for it in range(epochs):
        #     users = get_sample(data, 50)
        #     _sample = sample_Z(50, noise)
        #     _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss], feed_dict={network.discriminator_input: users, network.generator_input: _sample, network.generator_condition: users})
        #     _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss], feed_dict={network.generator_input: _sample, network.generator_condition: users})

        #     if it % 100 == 0:
        #         d_losses.append(D_loss_curr)
        #         g_losses.append(G_loss_curr)
        #         print('Iter: {}'.format(it))
        #         print('D loss: {:.4}'.format(D_loss_curr))
        #         print('G_loss: {:.4}'.format(G_loss_curr))
        #         d_losses.append(D_loss_curr)
        #         g_losses.append(G_loss_curr)
        #         print()

        # print("Saving model to {}".format(location))
        # saver.save(session, model_path)
        # plot_losses(int(epochs / 100), d_losses, g_losses)


if __name__ == '__main__':
    main(args)