#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf

from random import randint

from ganrecs.network import gan

from surprise import Dataset

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MOVIES_COUNT = 3706
USER_COUNT = 6040
TEST_PERCENT = .2

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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

    return location, int(args.noise), int(args.epochs)


def get_data():
    data = Dataset.load_builtin('ml-1m')
    user_tuples = {}
    movies = set([r[1] for r in data.raw_ratings])
    for user, movie, rating, _ in data.raw_ratings:
        if user not in user_tuples.keys():
            user_tuples[user] = {int(r):0 for r in movies}
        user_tuples[user][int(movie)] = float(rating) / 5.
    test = {}
    test_amount = int(USER_COUNT * TEST_PERCENT)
    keys = list(user_tuples.keys())
    for _ in range(test_amount):
        idx = randint(0, len(keys) - 1)
        usr = keys[idx]
        while usr in test.keys():
            idx = randint(0, len(keys) - 1)
            usr = keys[idx]
        test[usr] = user_tuples.pop(usr)
    return user_tuples, test


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


def main(args=None):
    location, noise, epochs = process_args(args)
    model_path = os.path.join(location, "model.ckpt")
    data, test_data = get_data()

    print("Constructing network...")
    dis_arch = [MOVIES_COUNT, 2000, 1000, 1]
    gen_arch = [noise, 3000, 2000, MOVIES_COUNT]
    network = gan(dis_arch, gen_arch, MOVIES_COUNT)

    saver = tf.train.Saver()

    session = tf.Session()
    if os.path.exists(model_path + ".meta"):
        print("Restoring model....")
        saver.restore(session, model_path)
    else:
        session.run(tf.global_variables_initializer())
        print("Starting run...")
        i = 0
        for it in range(epochs):
            users = get_sample(data, 50)
            _sample = sample_Z(50, noise)
            _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss], feed_dict={network.discriminator_input: users, network.generator_input: _sample, network.generator_condition: users})
            _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss], feed_dict={network.generator_input: _sample, network.generator_condition: users})

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
        
        print("Saving model to {}".format(location))
        saver.save(session, model_path)


if __name__ == '__main__':
    main(args)